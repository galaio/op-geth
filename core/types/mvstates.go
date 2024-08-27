package types

import (
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
	"unsafe"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/log"
	"github.com/holiman/uint256"
	"golang.org/x/exp/slices"
)

const (
	AccountStatePrefix = 'a'
	StorageStatePrefix = 's'
)

// RWKey max length is [1 + common.AddressLength + common.HashLength]byte
type RWKey string

type AccountState byte

const (
	AccountSelf AccountState = iota
	AccountNonce
	AccountBalance
	AccountCodeHash
	AccountSuicide
)

const (
	asyncDepGenChanSize = 1000
)

func AccountStateKey(account common.Address, state AccountState) RWKey {
	buf := make([]byte, 2+common.AddressLength)
	buf[0] = AccountStatePrefix
	copy(buf[1:], account.Bytes())
	buf[1+common.AddressLength] = byte(state)
	return RWKey(bytes2Str(buf))
}

func StorageStateKey(account common.Address, state common.Hash) RWKey {
	buf := make([]byte, 1+common.AddressLength+common.HashLength)
	buf[0] = StorageStatePrefix
	copy(buf[1:], account.Bytes())
	copy(buf[1+common.AddressLength:], state.Bytes())
	return RWKey(bytes2Str(buf))
}

func (key RWKey) IsAccountState() (bool, AccountState) {
	return AccountStatePrefix == key[0], AccountState(key[1+common.AddressLength])
}

func (key RWKey) IsAccountSelf() bool {
	ok, s := key.IsAccountState()
	if !ok {
		return false
	}
	return s == AccountSelf
}

func (key RWKey) IsAccountSuicide() bool {
	ok, s := key.IsAccountState()
	if !ok {
		return false
	}
	return s == AccountSuicide
}

func (key RWKey) ToAccountSelf() RWKey {
	return AccountStateKey(key.Addr(), AccountSelf)
}

func (key RWKey) IsStorageState() bool {
	return StorageStatePrefix == key[0]
}

func (key RWKey) String() string {
	return hex.EncodeToString([]byte(key))
}

func (key RWKey) Addr() common.Address {
	raw := str2Bytes(string(key))
	return common.BytesToAddress(raw[1 : 1+common.AddressLength])
}

// StateVersion record specific TxIndex & TxIncarnation
// if TxIndex equals to -1, it means the state read from DB.
type StateVersion struct {
	TxIndex int
	// Tx incarnation used for multi ver state
	TxIncarnation int
}

// RWSet record all read & write set in txs
// Attention: this is not a concurrent safety structure
type RWSet struct {
	ver      StateVersion
	readSet  map[RWKey]*RWItem
	writeSet map[RWKey]*RWItem

	// some flags
	rwRecordDone bool
	excludedTx   bool
}

func NewRWSet(ver StateVersion) *RWSet {
	return &RWSet{
		ver:      ver,
		readSet:  make(map[RWKey]*RWItem),
		writeSet: make(map[RWKey]*RWItem),
	}
}

func (s *RWSet) RecordRead(key RWKey, ver StateVersion, val interface{}) {
	// only record the first read version
	if _, exist := s.readSet[key]; exist {
		return
	}
	s.readSet[key] = &RWItem{
		Ver: ver,
		Val: val,
	}
}

func (s *RWSet) RecordWrite(key RWKey, val interface{}) {
	wr, exist := s.writeSet[key]
	if !exist {
		s.writeSet[key] = &RWItem{
			Ver: s.ver,
			Val: val,
		}
		return
	}
	wr.Val = val
}

func (s *RWSet) Version() StateVersion {
	return s.ver
}

func (s *RWSet) ReadSet() map[RWKey]*RWItem {
	return s.readSet
}

func (s *RWSet) WriteSet() map[RWKey]*RWItem {
	return s.writeSet
}

func (s *RWSet) WithExcludedTxFlag() *RWSet {
	s.excludedTx = true
	return s
}

func (s *RWSet) String() string {
	builder := strings.Builder{}
	builder.WriteString(fmt.Sprintf("tx: %v, inc: %v\nreadSet: [", s.ver.TxIndex, s.ver.TxIncarnation))
	i := 0
	for key, _ := range s.readSet {
		if i > 0 {
			builder.WriteString(fmt.Sprintf(", %v", key.String()))
			continue
		}
		builder.WriteString(fmt.Sprintf("%v", key.String()))
		i++
	}
	builder.WriteString("]\nwriteSet: [")
	i = 0
	for key, _ := range s.writeSet {
		if i > 0 {
			builder.WriteString(fmt.Sprintf(", %v", key.String()))
			continue
		}
		builder.WriteString(fmt.Sprintf("%v", key.String()))
		i++
	}
	builder.WriteString("]\n")
	return builder.String()
}

// isEqualRWVal compare state
func isEqualRWVal(key RWKey, src interface{}, compared interface{}) bool {
	if ok, state := key.IsAccountState(); ok {
		switch state {
		case AccountBalance:
			if src != nil && compared != nil {
				return equalUint256(src.(*uint256.Int), compared.(*uint256.Int))
			}
			return src == compared
		case AccountNonce:
			return src.(uint64) == compared.(uint64)
		case AccountCodeHash:
			if src != nil && compared != nil {
				return slices.Equal(src.([]byte), compared.([]byte))
			}
			return src == compared
		}
		return false
	}

	if src != nil && compared != nil {
		return src.(common.Hash) == compared.(common.Hash)
	}
	return src == compared
}

func equalUint256(s, c *uint256.Int) bool {
	if s != nil && c != nil {
		return s.Eq(c)
	}

	return s == c
}

type RWItem struct {
	Ver StateVersion
	Val interface{}
}

func NewRWItem(ver StateVersion, val interface{}) *RWItem {
	return &RWItem{
		Ver: ver,
		Val: val,
	}
}

func (w *RWItem) TxIndex() int {
	return w.Ver.TxIndex
}

func (w *RWItem) TxIncarnation() int {
	return w.Ver.TxIncarnation
}

type PendingWrites struct {
	list []*RWItem
}

func NewPendingWrites() *PendingWrites {
	return &PendingWrites{
		list: make([]*RWItem, 0),
	}
}

func (w *PendingWrites) Append(pw *RWItem) {
	if i, found := w.SearchTxIndex(pw.TxIndex()); found {
		w.list[i] = pw
		return
	}

	w.list = append(w.list, pw)
	for i := len(w.list) - 1; i > 0; i-- {
		if w.list[i].TxIndex() > w.list[i-1].TxIndex() {
			break
		}
		w.list[i-1], w.list[i] = w.list[i], w.list[i-1]
	}
}

func (w *PendingWrites) SearchTxIndex(txIndex int) (int, bool) {
	n := len(w.list)
	i, j := 0, n
	for i < j {
		h := int(uint(i+j) >> 1)
		// i ≤ h < j
		if w.list[h].TxIndex() < txIndex {
			i = h + 1
		} else {
			j = h
		}
	}
	return i, i < n && w.list[i].TxIndex() == txIndex
}

func (w *PendingWrites) FindLastWrite(txIndex int) *RWItem {
	var i, _ = w.SearchTxIndex(txIndex)
	for j := i - 1; j >= 0; j-- {
		if w.list[j].TxIndex() < txIndex {
			return w.list[j]
		}
	}

	return nil
}

func (w *PendingWrites) FindPrevWrites(txIndex int) []*RWItem {
	var i, _ = w.SearchTxIndex(txIndex)
	for j := i - 1; j >= 0; j-- {
		if w.list[j].TxIndex() < txIndex {
			return w.list[:j+1]
		}
	}

	return nil
}

type MVStates struct {
	rwSets            map[int]*RWSet
	pendingWriteSet   map[RWKey]*PendingWrites
	nextFinaliseIndex int

	// dependency map cache for generating TxDAG
	// depMapCache[i].exist(j) means j->i, and i > j
	txDepCache []TxDepMaker

	// async dep analysis
	asyncGenChan  chan int
	asyncStopChan chan struct{}
	asyncRunning  bool

	// execution stat infos
	stats map[int]*ExeStat
	lock  sync.RWMutex
}

func NewMVStates(txCount int) *MVStates {
	return &MVStates{
		rwSets:          make(map[int]*RWSet, txCount),
		pendingWriteSet: make(map[RWKey]*PendingWrites, txCount*8),
		txDepCache:      make([]TxDepMaker, 0, txCount),
		stats:           make(map[int]*ExeStat, txCount),
	}
}

func (s *MVStates) EnableAsyncGen() *MVStates {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.asyncGenChan = make(chan int, asyncDepGenChanSize)
	s.asyncStopChan = make(chan struct{})
	s.asyncRunning = true
	go s.asyncGenLoop()
	return s
}

func (s *MVStates) Stop() error {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.stopAsyncGen()
	return nil
}

func (s *MVStates) stopAsyncGen() {
	if !s.asyncRunning {
		return
	}
	s.asyncRunning = false
	if s.asyncStopChan != nil {
		close(s.asyncStopChan)
	}
}

func (s *MVStates) asyncGenLoop() {
	for {
		select {
		case tx := <-s.asyncGenChan:
			s.Finalise(tx)
		case <-s.asyncStopChan:
			return
		}
	}
}

func (s *MVStates) RWSets() map[int]*RWSet {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.rwSets
}

func (s *MVStates) Stats() map[int]*ExeStat {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.stats
}

func (s *MVStates) RWSet(index int) *RWSet {
	s.lock.RLock()
	defer s.lock.RUnlock()
	if index >= len(s.rwSets) {
		return nil
	}
	return s.rwSets[index]
}

// ReadState read state from MVStates
func (s *MVStates) ReadState(txIndex int, key RWKey) *RWItem {
	s.lock.RLock()
	defer s.lock.RUnlock()

	wset, ok := s.pendingWriteSet[key]
	if !ok {
		return nil
	}
	return wset.FindLastWrite(txIndex)
}

// FulfillRWSet it can execute as async, and rwSet & stat must guarantee read-only
// try to generate TxDAG, when fulfill RWSet
func (s *MVStates) FulfillRWSet(rwSet *RWSet, stat *ExeStat) error {
	s.lock.Lock()
	defer s.lock.Unlock()
	index := rwSet.ver.TxIndex
	if index < s.nextFinaliseIndex {
		return errors.New("fulfill a finalized RWSet")
	}
	if stat != nil {
		if stat.txIndex != index {
			return errors.New("wrong execution stat")
		}
		s.stats[index] = stat
	}

	//if metrics.EnabledExpensive {
	//	for k := range rwSet.writeSet {
	//		// this action is only for testing, it runs when enable expensive metrics.
	//		checkRWSetInconsistent(index, k, rwSet.readSet, rwSet.writeSet)
	//	}
	//}
	s.rwSets[index] = rwSet
	return nil
}

// AsyncFinalise it will put target write set into pending writes.
func (s *MVStates) AsyncFinalise(index int) {
	// async resolve dependency, but non-block action
	if s.asyncRunning && s.asyncGenChan != nil {
		s.asyncGenChan <- index
	}
}

// Finalise it will put target write set into pending writes.
func (s *MVStates) Finalise(index int) error {
	s.lock.Lock()
	defer s.lock.Unlock()
	if err := s.innerFinalise(index); err != nil {
		return err
	}

	s.resolveDepsCacheByWrites(index, s.rwSets[index])
	return nil
}

func (s *MVStates) innerFinalise(index int) error {
	rwSet := s.rwSets[index]
	if rwSet == nil {
		return fmt.Errorf("finalise a non-exist RWSet, index: %d", index)
	}

	if index != s.nextFinaliseIndex {
		return fmt.Errorf("finalise in wrong order, next: %d, input: %d", s.nextFinaliseIndex, index)
	}

	// append to pending write set
	for k, v := range rwSet.writeSet {
		if _, exist := s.pendingWriteSet[k]; !exist {
			s.pendingWriteSet[k] = NewPendingWrites()
		}
		s.pendingWriteSet[k].Append(v)
	}
	s.nextFinaliseIndex++
	return nil
}

// resolveDepsCacheByWrites must be executed in order
func (s *MVStates) resolveDepsCacheByWrites(index int, rwSet *RWSet) {
	// analysis dep, if the previous transaction is not executed/validated, re-analysis is required
	depSlice := NewTxDepSlice(0)
	if rwSet.excludedTx {
		s.txDepCache = append(s.txDepCache, depSlice)
		return
	}
	// check tx dependency, only check key, skip version
	if len(s.pendingWriteSet) > len(rwSet.readSet) {
		for key := range rwSet.readSet {
			// check self destruct
			if key.IsAccountSelf() {
				key = AccountStateKey(key.Addr(), AccountSuicide)
			}
			writes := s.pendingWriteSet[key]
			if writes == nil {
				continue
			}
			items := writes.FindPrevWrites(index)
			for _, item := range items {
				depSlice.add(uint64(item.TxIndex()))
			}
		}
	} else {
		for k, w := range s.pendingWriteSet {
			// check suicide, add read address flag, it only for check suicide quickly, and cannot for other scenarios.
			if k.IsAccountSuicide() {
				k = k.ToAccountSelf()
			}
			if _, ok := rwSet.readSet[k]; !ok {
				continue
			}
			items := w.FindPrevWrites(index)
			for _, item := range items {
				depSlice.add(uint64(item.TxIndex()))
			}
		}
	}
	if depSlice.len() == 0 {
		s.txDepCache = append(s.txDepCache, depSlice)
		return
	}
	// clear redundancy deps compared with prev
	delSlice := NewTxDepSlice(depSlice.len())
	for _, prev := range depSlice.deps() {
		for _, tx := range s.txDepCache[prev].deps() {
			if depSlice.exist(tx) {
				delSlice.add(tx)
			}
		}
	}
	if delSlice.len() == 0 {
		s.txDepCache = append(s.txDepCache, depSlice)
		return
	}

	nSlice := NewTxDepSlice(depSlice.len() - delSlice.len())
	for _, tx := range depSlice.deps() {
		if delSlice.exist(tx) {
			continue
		}
		nSlice.add(tx)
	}
	s.txDepCache = append(s.txDepCache, nSlice)
}

// resolveDepsCache must be executed in order
func (s *MVStates) resolveDepsCache(index int, rwSet *RWSet) {
	// analysis dep, if the previous transaction is not executed/validated, re-analysis is required
	depMap := NewTxDepMap(0)
	if rwSet.excludedTx {
		s.txDepCache = append(s.txDepCache, depMap)
		return
	}
	for prev := 0; prev < index; prev++ {
		// if there are some parallel execution or system txs, it will fulfill in advance
		// it's ok, and try re-generate later
		prevSet, ok := s.rwSets[prev]
		if !ok {
			continue
		}
		// if prev tx is tagged ExcludedTxFlag, just skip the check
		if prevSet.excludedTx {
			continue
		}
		// check if there has written op before i
		if checkDependency(prevSet.writeSet, rwSet.readSet) {
			depMap.add(uint64(prev))
			// clear redundancy deps compared with prev
			for _, dep := range depMap.deps() {
				if s.txDepCache[prev].exist(dep) {
					depMap.remove(dep)
				}
			}
		}
	}
	s.txDepCache = append(s.txDepCache, depMap)
}

// checkRWSetInconsistent a helper function
func checkRWSetInconsistent(index int, k RWKey, readSet map[RWKey]*RWItem, writeSet map[RWKey]*RWItem) bool {
	var (
		readOk  bool
		writeOk bool
		r       *RWItem
	)

	if k.IsAccountSuicide() {
		_, readOk = readSet[k.ToAccountSelf()]
	} else {
		_, readOk = readSet[k]
	}

	r, writeOk = writeSet[k]
	if readOk != writeOk {
		// check if it's correct? read nil, write non-nil
		log.Warn("checkRWSetInconsistent find inconsistent", "tx", index, "k", k.String(), "read", readOk, "write", writeOk, "val", r.Val)
		return true
	}

	return false
}

// ResolveTxDAG generate TxDAG from RWSets
func (s *MVStates) ResolveTxDAG(txCnt int, gasFeeReceivers []common.Address) (TxDAG, error) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if len(s.rwSets) != txCnt {
		return nil, fmt.Errorf("wrong rwSet count, expect: %v, actual: %v", txCnt, len(s.rwSets))
	}
	s.stopAsyncGen()
	// collect all rw sets, try to finalise them
	for i := s.nextFinaliseIndex; i < txCnt; i++ {
		if err := s.innerFinalise(i); err != nil {
			return nil, err
		}
	}

	txDAG := NewPlainTxDAG(txCnt)
	for i := 0; i < txCnt; i++ {
		// check if there are RW with gas fee receiver for gas delay calculation
		for _, addr := range gasFeeReceivers {
			if _, ok := s.rwSets[i].readSet[AccountStateKey(addr, AccountSelf)]; ok {
				return NewEmptyTxDAG(), nil
			}
		}
		txDAG.TxDeps[i].TxIndexes = []uint64{}
		if len(s.txDepCache) <= i {
			s.resolveDepsCacheByWrites(i, s.rwSets[i])
		}
		if s.rwSets[i].excludedTx {
			txDAG.TxDeps[i].SetFlag(ExcludedTxFlag)
			continue
		}
		deps := s.txDepCache[i].deps()
		if len(deps) <= (txCnt-1)/2 {
			txDAG.TxDeps[i].TxIndexes = deps
			continue
		}
		// if tx deps larger than half of txs, then convert with NonDependentRelFlag
		txDAG.TxDeps[i].SetFlag(NonDependentRelFlag)
		for j := uint64(0); j < uint64(txCnt); j++ {
			if !slices.Contains(deps, j) && j != uint64(i) {
				txDAG.TxDeps[i].TxIndexes = append(txDAG.TxDeps[i].TxIndexes, j)
			}
		}
	}

	return txDAG, nil
}

func checkDependency(writeSet map[RWKey]*RWItem, readSet map[RWKey]*RWItem) bool {
	// check tx dependency, only check key, skip version
	for k, _ := range writeSet {
		// check suicide, add read address flag, it only for check suicide quickly, and cannot for other scenarios.
		if k.IsAccountSuicide() {
			if _, ok := readSet[k.ToAccountSelf()]; ok {
				return true
			}
			continue
		}
		if _, ok := readSet[k]; ok {
			return true
		}
	}

	return false
}

type TxDepMaker interface {
	add(index uint64)
	exist(index uint64) bool
	deps() []uint64
	remove(index uint64)
	len() int
}

type TxDepSlice struct {
	indexes []uint64
}

func NewTxDepSlice(cap int) *TxDepSlice {
	return &TxDepSlice{
		indexes: make([]uint64, 0, cap),
	}
}

func (m *TxDepSlice) add(index uint64) {
	if m.exist(index) {
		return
	}
	m.indexes = append(m.indexes, index)
	for i := len(m.indexes) - 1; i > 0; i-- {
		if m.indexes[i] < m.indexes[i-1] {
			m.indexes[i-1], m.indexes[i] = m.indexes[i], m.indexes[i-1]
		}
	}
}

func (m *TxDepSlice) exist(index uint64) bool {
	_, ok := slices.BinarySearch(m.indexes, index)
	return ok
}

func (m *TxDepSlice) deps() []uint64 {
	return m.indexes
}

func (m *TxDepSlice) remove(index uint64) {
	pos, ok := slices.BinarySearch(m.indexes, index)
	if !ok {
		return
	}
	for i := pos; i < len(m.indexes)-1; i++ {
		m.indexes[i] = m.indexes[i+1]
	}
	m.indexes = m.indexes[:len(m.indexes)-1]
}

func (m *TxDepSlice) len() int {
	return len(m.indexes)
}

type TxDepMap map[uint64]struct{}

func NewTxDepMap(cap int) TxDepMap {
	return make(map[uint64]struct{}, cap)
}

func (m TxDepMap) add(index uint64) {
	m[index] = struct{}{}
}

func (m TxDepMap) exist(index uint64) bool {
	_, ok := m[index]
	return ok
}

func (m TxDepMap) deps() []uint64 {
	ret := make([]uint64, 0, len(m))
	for index := range m {
		ret = append(ret, uint64(index))
	}
	slices.Sort(ret)
	return ret
}

func (m TxDepMap) remove(index uint64) {
	delete(m, index)
}

func (m TxDepMap) len() int {
	return len(m)
}

func bytes2Str(buf []byte) string {
	p := unsafe.SliceData(buf)
	return unsafe.String(p, len(buf))
}

func str2Bytes(s string) []byte {
	p := unsafe.StringData(s)
	return unsafe.Slice(p, len(s))
}
