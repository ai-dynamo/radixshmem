// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// radix_etcd.go — minimal cgo bridge from RadixShmem (C++) to the official Go
// etcd client (go.etcd.io/etcd/client/v3). Built as a c-shared library:
//
//   go build -buildmode=c-shared -o libradix_etcd.so radix_etcd.go
//
// which also emits libradix_etcd.h (the C surface below). The C++ EtcdRegistry
// links this .so and calls these functions; all grpc/protobuf machinery is
// statically bundled into the .so, so the C++ build needs no etcd C++ deps.
//
// Convention: every call returns 0 on success, negative on error. On error the
// call writes a malloc'd C string into *errMsg (caller frees). Values are opaque
// byte strings passed with explicit sizes (they may hold base64 or raw bytes).
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	clientv3 "go.etcd.io/etcd/client/v3"
)

var (
	cli   *clientv3.Client
	cliMu sync.Mutex
)

// A watch event delivered from a background goroutine to the C++ consumer.
type watchEvent struct {
	typ int // 0 = PUT, 1 = DELETE
	key []byte
	val []byte
}

type watchEntry struct {
	ch       chan watchEvent
	cancel   context.CancelFunc
	overflow int32 // set when ch was full and an event was dropped
}

var (
	watches     = map[int64]*watchEntry{}
	watchMu     sync.Mutex
	nextWatchID int64
)

const watchChanCap = 256

var (
	leases   = map[int64]context.CancelFunc{}
	leasesMu sync.Mutex
)

func fail(errMsg **C.char, msg string) C.int {
	*errMsg = C.CString(msg)
	return -1
}

// RadixEtcdInit — dial etcd; endpoints comma/semicolon separated (host:port,
// with or without http:// scheme). Idempotent: a second call is a no-op.
//
//export RadixEtcdInit
func RadixEtcdInit(endpoints *C.char, errMsg **C.char) C.int {
	cliMu.Lock()
	defer cliMu.Unlock()
	if cli != nil {
		return 0
	}
	raw := strings.ReplaceAll(C.GoString(endpoints), ",", ";")
	var eps []string
	for _, e := range strings.Split(raw, ";") {
		if e = strings.TrimSpace(e); e != "" {
			eps = append(eps, e)
		}
	}
	if len(eps) == 0 {
		return fail(errMsg, "no valid endpoints")
	}
	c, err := clientv3.New(clientv3.Config{
		Endpoints:   eps,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return fail(errMsg, err.Error())
	}
	cli = c
	return 0
}

// RadixEtcdClose — tear down the client.
//
//export RadixEtcdClose
func RadixEtcdClose() {
	leasesMu.Lock()
	for id, cancel := range leases {
		cancel()
		delete(leases, id)
	}
	leasesMu.Unlock()
	cliMu.Lock()
	defer cliMu.Unlock()
	if cli != nil {
		cli.Close()
		cli = nil
	}
}

// RadixEtcdKeepAlive — renew leaseId until RadixEtcdRevoke or RadixEtcdClose.
//
//export RadixEtcdKeepAlive
func RadixEtcdKeepAlive(leaseId C.longlong, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := cli.KeepAlive(ctx, clientv3.LeaseID(leaseId))
	if err != nil {
		cancel()
		return fail(errMsg, err.Error())
	}
	leasesMu.Lock()
	leases[int64(leaseId)] = cancel
	leasesMu.Unlock()
	go func() {
		for range ch {
		}
	}()
	return 0
}

// RadixEtcdKeepAliveStop — stop renewing leaseId without revoking it; its keys
// live on until the lease expires or another holder keeps renewing it.
//
//export RadixEtcdKeepAliveStop
func RadixEtcdKeepAliveStop(leaseId C.longlong) {
	leasesMu.Lock()
	if cancel := leases[int64(leaseId)]; cancel != nil {
		cancel()
		delete(leases, int64(leaseId))
	}
	leasesMu.Unlock()
}

// RadixEtcdRevoke — stop renewing leaseId and revoke it; its keys vanish at once.
//
//export RadixEtcdRevoke
func RadixEtcdRevoke(leaseId C.longlong, errMsg **C.char) C.int {
	leasesMu.Lock()
	if cancel := leases[int64(leaseId)]; cancel != nil {
		cancel()
		delete(leases, int64(leaseId))
	}
	leasesMu.Unlock()
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := cli.Revoke(ctx, clientv3.LeaseID(leaseId)); err != nil {
		return fail(errMsg, err.Error())
	}
	return 0
}

// RadixEtcdGrantLease — grant a lease with ttl seconds; returns id in *leaseId.
//
//export RadixEtcdGrantLease
func RadixEtcdGrantLease(ttl C.longlong, leaseId *C.longlong, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := cli.Grant(ctx, int64(ttl))
	if err != nil {
		return fail(errMsg, err.Error())
	}
	*leaseId = C.longlong(resp.ID)
	return 0
}

// RadixEtcdPut — unconditional put of value[0:valueSize] at key (null-terminated).
// leaseId <= 0 means no lease.
//
//export RadixEtcdPut
func RadixEtcdPut(key *C.char, value *C.char, valueSize C.int, leaseId C.longlong, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	k := C.GoString(key)
	v := C.GoStringN(value, valueSize)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var err error
	if leaseId > 0 {
		_, err = cli.Put(ctx, k, v, clientv3.WithLease(clientv3.LeaseID(leaseId)))
	} else {
		_, err = cli.Put(ctx, k, v)
	}
	if err != nil {
		return fail(errMsg, err.Error())
	}
	return 0
}

// RadixEtcdCreate — create-if-absent (leader election / barrier claim). Puts key
// only if it has no prior create-revision. *created = 1 if we created it, 0 if it
// already existed. leaseId <= 0 means no lease.
//
//export RadixEtcdCreate
func RadixEtcdCreate(key *C.char, value *C.char, valueSize C.int, leaseId C.longlong, created *C.int, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	k := C.GoString(key)
	v := C.GoStringN(value, valueSize)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var put clientv3.Op
	if leaseId > 0 {
		put = clientv3.OpPut(k, v, clientv3.WithLease(clientv3.LeaseID(leaseId)))
	} else {
		put = clientv3.OpPut(k, v)
	}
	resp, err := cli.Txn(ctx).
		If(clientv3.Compare(clientv3.CreateRevision(k), "=", 0)).
		Then(put).
		Commit()
	if err != nil {
		return fail(errMsg, err.Error())
	}
	if resp.Succeeded {
		*created = 1
	} else {
		*created = 0
	}
	return 0
}

// RadixEtcdGet — get a single key. If found, *found=1 and *value is a malloc'd
// buffer of *valueSize bytes (caller frees). If absent, *found=0.
//
//export RadixEtcdGet
func RadixEtcdGet(key *C.char, found *C.int, value **C.char, valueSize *C.int, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	k := C.GoString(key)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := cli.Get(ctx, k)
	if err != nil {
		return fail(errMsg, err.Error())
	}
	if len(resp.Kvs) == 0 {
		*found = 0
		return 0
	}
	*found = 1
	b := resp.Kvs[0].Value
	*value = (*C.char)(C.CBytes(b))
	*valueSize = C.int(len(b))
	return 0
}

// RadixEtcdGetWithLease — RadixEtcdGet that also returns the key's lease id in
// *leaseId (0 when the key has no lease or is absent).
//
//export RadixEtcdGetWithLease
func RadixEtcdGetWithLease(key *C.char, found *C.int, value **C.char, valueSize *C.int,
	leaseId *C.longlong, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	k := C.GoString(key)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := cli.Get(ctx, k)
	if err != nil {
		return fail(errMsg, err.Error())
	}
	*leaseId = 0
	if len(resp.Kvs) == 0 {
		*found = 0
		return 0
	}
	*found = 1
	b := resp.Kvs[0].Value
	*value = (*C.char)(C.CBytes(b))
	*valueSize = C.int(len(b))
	*leaseId = C.longlong(resp.Kvs[0].Lease)
	return 0
}

// RadixEtcdCount — number of keys under a prefix (cheap barrier poll). Result in
// *count. Uses WithCountOnly so no values transfer.
//
//export RadixEtcdCount
func RadixEtcdCount(prefix *C.char, count *C.longlong, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	p := C.GoString(prefix)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := cli.Get(ctx, p, clientv3.WithPrefix(), clientv3.WithCountOnly())
	if err != nil {
		return fail(errMsg, err.Error())
	}
	*count = C.longlong(resp.Count)
	return 0
}

// RadixEtcdGetPrefix — get all key/value pairs under a prefix, sorted by key
// ascending. On success *count entries are returned via four malloc'd arrays:
// keys (char*), keySizes (int), values (char*), valueSizes (int). Each keys[i] /
// values[i] is an individually malloc'd buffer. Caller frees every element then
// the four arrays. When *count==0 the array pointers are left NULL. *revision
// receives the snapshot revision, for anchoring a watch.
//
//export RadixEtcdGetPrefix
func RadixEtcdGetPrefix(prefix *C.char, count *C.int,
	keys ***C.char, keySizes **C.int, values ***C.char, valueSizes **C.int,
	revision *C.longlong, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	p := C.GoString(prefix)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := cli.Get(ctx, p, clientv3.WithPrefix(),
		clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend))
	if err != nil {
		return fail(errMsg, err.Error())
	}
	if revision != nil {
		*revision = C.longlong(resp.Header.Revision)
	}
	n := len(resp.Kvs)
	*count = C.int(n)
	if n == 0 {
		return 0
	}
	ptrSize := C.size_t(unsafe.Sizeof((*C.char)(nil)))
	intSize := C.size_t(unsafe.Sizeof(C.int(0)))
	kArr := (*[1 << 28]*C.char)(C.malloc(C.size_t(n) * ptrSize))
	kSz := (*[1 << 28]C.int)(C.malloc(C.size_t(n) * intSize))
	vArr := (*[1 << 28]*C.char)(C.malloc(C.size_t(n) * ptrSize))
	vSz := (*[1 << 28]C.int)(C.malloc(C.size_t(n) * intSize))
	for i, kv := range resp.Kvs {
		kArr[i] = (*C.char)(C.CBytes(kv.Key))
		kSz[i] = C.int(len(kv.Key))
		vArr[i] = (*C.char)(C.CBytes(kv.Value))
		vSz[i] = C.int(len(kv.Value))
	}
	*keys = (**C.char)(unsafe.Pointer(kArr))
	*keySizes = (*C.int)(unsafe.Pointer(kSz))
	*values = (**C.char)(unsafe.Pointer(vArr))
	*valueSizes = (*C.int)(unsafe.Pointer(vSz))
	return 0
}

// RadixEtcdDelete — delete exactly one key. Absent key is not an error.
//
//export RadixEtcdDelete
func RadixEtcdDelete(key *C.char, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	k := C.GoString(key)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := cli.Delete(ctx, k); err != nil {
		return fail(errMsg, err.Error())
	}
	return 0
}

// RadixEtcdDeletePrefix — delete every key under a prefix (test teardown).
//
//export RadixEtcdDeletePrefix
func RadixEtcdDeletePrefix(prefix *C.char, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	p := C.GoString(prefix)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := cli.Delete(ctx, p, clientv3.WithPrefix()); err != nil {
		return fail(errMsg, err.Error())
	}
	return 0
}

// RadixEtcdWatchStart — subscribe to a key prefix. A background goroutine streams
// etcd events into a buffered channel; the C++ side drains it via WatchNext. On
// success *watchId identifies the watch. startRev == 0 delivers from the current
// revision; > 0 delivers from that revision inclusive.
//
//export RadixEtcdWatchStart
func RadixEtcdWatchStart(prefix *C.char, startRev C.longlong,
	watchId *C.longlong, errMsg **C.char) C.int {
	if cli == nil {
		return fail(errMsg, "etcd not initialized")
	}
	p := C.GoString(prefix)
	ctx, cancel := context.WithCancel(context.Background())
	entry := &watchEntry{ch: make(chan watchEvent, watchChanCap), cancel: cancel}

	watchMu.Lock()
	nextWatchID++
	id := nextWatchID
	watches[id] = entry
	watchMu.Unlock()

	opts := []clientv3.OpOption{clientv3.WithPrefix()}
	if startRev > 0 {
		opts = append(opts, clientv3.WithRev(int64(startRev)))
	}
	wch := cli.Watch(ctx, p, opts...)
	go func() {
		for wresp := range wch {
			for _, ev := range wresp.Events {
				typ := 0
				if ev.Type == clientv3.EventTypeDelete {
					typ = 1
				}
				e := watchEvent{typ: typ, key: ev.Kv.Key, val: ev.Kv.Value}
				select {
				case entry.ch <- e:
				default:
					atomic.StoreInt32(&entry.overflow, 1) // full: drop, flag for reconcile
				}
			}
		}
	}()

	*watchId = C.longlong(id)
	return 0
}

// RadixEtcdWatchNext — take one event from a watch's channel. timeoutMs < 0 blocks
// until an event arrives; == 0 polls; > 0 waits up to the deadline. *hasEvent is 1
// when an event is returned (then *evType/*key/*value + sizes are filled, buffers
// malloc'd for the caller to free), else 0. *evType: 0 = PUT, 1 = DELETE.
//
//export RadixEtcdWatchNext
func RadixEtcdWatchNext(watchId C.longlong, timeoutMs C.int,
	hasEvent *C.int, evType *C.int,
	key **C.char, keySize *C.int, value **C.char, valueSize *C.int,
	errMsg **C.char) C.int {

	watchMu.Lock()
	entry := watches[int64(watchId)]
	watchMu.Unlock()
	if entry == nil {
		return fail(errMsg, "unknown watch id")
	}

	var ev watchEvent
	got := false
	if timeoutMs < 0 {
		ev, got = <-entry.ch, true
	} else if timeoutMs == 0 {
		select {
		case ev = <-entry.ch:
			got = true
		default:
		}
	} else {
		select {
		case ev = <-entry.ch:
			got = true
		case <-time.After(time.Duration(timeoutMs) * time.Millisecond):
		}
	}

	if !got {
		*hasEvent = 0
		return 0
	}
	*hasEvent = 1
	*evType = C.int(ev.typ)
	*key = (*C.char)(C.CBytes(ev.key))
	*keySize = C.int(len(ev.key))
	*value = (*C.char)(C.CBytes(ev.val))
	*valueSize = C.int(len(ev.val))
	return 0
}

// RadixEtcdWatchOverflow — read-and-clear the watch's dropped-event flag. Returns
// *flag=1 if the channel overflowed since the last call (the consumer should then
// reconcile via a full prefix scan), else 0.
//
//export RadixEtcdWatchOverflow
func RadixEtcdWatchOverflow(watchId C.longlong, flag *C.int, errMsg **C.char) C.int {
	watchMu.Lock()
	entry := watches[int64(watchId)]
	watchMu.Unlock()
	if entry == nil {
		return fail(errMsg, "unknown watch id")
	}
	*flag = C.int(atomic.SwapInt32(&entry.overflow, 0))
	return 0
}

// RadixEtcdWatchStop — cancel a watch's goroutine and drop its table entry.
// Idempotent: stopping an unknown/already-stopped id is a no-op success.
//
//export RadixEtcdWatchStop
func RadixEtcdWatchStop(watchId C.longlong, errMsg **C.char) C.int {
	watchMu.Lock()
	entry := watches[int64(watchId)]
	delete(watches, int64(watchId))
	watchMu.Unlock()
	if entry != nil {
		entry.cancel()
	}
	return 0
}

func main() {}
