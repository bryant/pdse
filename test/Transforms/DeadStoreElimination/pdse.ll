; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -basicaa -pdse %s | FileCheck %s

declare void @may_throw()
declare noalias i8* @malloc(i32)
declare void @free(i8* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64 , i32 , i1 )
declare void @llvm.memmove.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i32, i1)
declare i32 @personality(...)
declare i64 @f(i64*)
declare void @llvm.lifetime.end(i64, i8*)
declare void @llvm.lifetime.start(i64, i8*)

; Example from Figure 10 of https://doi.org/10.1145/277650.277659 . Note how the
; bb3 loop-invariant store is sunk out of the loop.
define void @lo_and_chow(i8* %x, i1 %br0, i1 %br1) {
; CHECK-LABEL: @lo_and_chow(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[V:%.*]] = load i8, i8* [[X:%.*]]
; CHECK-NEXT:    [[V1:%.*]] = add nuw i8 [[V]], 1
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP0:%.*]] = phi i8* [ [[X]], [[BB3:%.*]] ], [ [[X]], [[BB0:%.*]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = phi i8 [ [[V1]], [[BB3]] ], [ [[V1]], [[BB0]] ]
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB2:%.*]], label [[BB3]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[TMP0]]
; CHECK-NEXT:    [[T:%.*]] = load i8, i8* [[X]]
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    br i1 [[BR1:%.*]], label [[BB1]], label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    store i8 [[V1]], i8* [[X]]
; CHECK-NEXT:    ret void
;
bb0:
  %v = load i8, i8* %x
  %v1 = add nuw i8 %v, 1
  store i8 %v1, i8* %x
  br label %bb1
bb1:
  br i1 %br0, label %bb2, label %bb3
bb2:
  %t = load i8, i8* %x
  br label %bb3
bb3:
  store i8 %v1, i8* %x
  br i1 %br1, label %bb1, label %exit
exit:
  ret void
}

; Same as above, but with a kill occurrence by a may-throw call.
define void @lo_and_chow_maythrow(i8* %x, i1 %br0, i1 %br1) {
; CHECK-LABEL: @lo_and_chow_maythrow(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[V:%.*]] = load i8, i8* [[X:%.*]]
; CHECK-NEXT:    [[V1:%.*]] = add nuw i8 [[V]], 1
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP0:%.*]] = phi i8* [ [[X]], [[BB3:%.*]] ], [ [[X]], [[BB0:%.*]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = phi i8 [ [[V1]], [[BB3]] ], [ [[V1]], [[BB0]] ]
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB2:%.*]], label [[BB3]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[TMP0]]
; CHECK-NEXT:    call void @may_throw()
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    br i1 [[BR1:%.*]], label [[BB1]], label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    store i8 [[V1]], i8* [[X]]
; CHECK-NEXT:    ret void
;
bb0:
  %v = load i8, i8* %x
  %v1 = add nuw i8 %v, 1
  store i8 %v1, i8* %x
  br label %bb1
bb1:
  br i1 %br0, label %bb2, label %bb3
bb2:
  call void @may_throw()
  br label %bb3
bb3:
  store i8 %v1, i8* %x
  br i1 %br1, label %bb1, label %exit
exit:
  ret void
}

; demos the self-loop problem in post-dom tree.
; define void @f(i8* %x) {
; a:
;     store i8 0, i8* %x
;     switch i8 0, label %b [
;         i8 1, label %c
;     ]
; b:
;     store i8 1, i8* %x
;     br label %b
; c:
;     store i8 2, i8* %x
;     br label %d
; d:
;     br label %d
; e:
;     store i8 3, i8* %x
;     ret void
; }
;
; define void @g(i8* %a, i8* %b) {
; bb0:
;     store i8 undef, i8* %b
;     store i8 undef, i8* %a
;     br i1 undef, label %bb1, label %bb2
; bb1:
;     %tmp0 = load i8, i8* %a
;     ret void
; bb2:
;     store i8 undef, i8* %a
;     ret void
; }

; define void @i(i8* noalias %x, i8* noalias %y, i1 %z) {
;     %whatever = load i8, i8* %x
;     br label %nextblock
;
; nextblock:
;     store i8 %whatever, i8* %x
;     store i8 123, i8* %x
;     br i1 %z, label %nextblock, label %fin
;
; fin:
;     ret void
; }

define i8* @j(i8* %a, i8* %e, i1 %c) {
; CHECK-LABEL: @j(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[P:%.*]] = tail call i8* @malloc(i32 4)
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    call void @llvm.memmove.p0i8.p0i8.i64(i8* [[A:%.*]], i8* nonnull [[E:%.*]], i64 64, i32 8, i1 false)
; CHECK-NEXT:    call void @llvm.memset.p0i8.i64(i8* [[A]], i8 undef, i64 32, i32 8, i1 false)
; CHECK-NEXT:    [[X:%.*]] = bitcast i8* [[A]] to i64*
; CHECK-NEXT:    [[Z:%.*]] = getelementptr i64, i64* [[X]], i64 1
; CHECK-NEXT:    store i64 undef, i64* [[Z]]
; CHECK-NEXT:    store i8 undef, i8* [[A]]
; CHECK-NEXT:    call void @may_throw()
; CHECK-NEXT:    store i8 0, i8* [[P]]
; CHECK-NEXT:    store i8 undef, i8* [[A]]
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    ret i8* [[P]]
;
bb0:
  %b = alloca i8
  %P = tail call i8* @malloc(i32 4)
  br i1 %c, label %bb1, label %bb2
bb1:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %b, i8* nonnull %a, i64 64, i32 8, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %a, i8* nonnull %e, i64 64, i32 8, i1 false)
  call void @llvm.memset.p0i8.i64(i8* %a, i8 undef, i64 32, i32 8, i1 false)
  %x = bitcast i8* %a to i64*
  %z = getelementptr i64, i64* %x, i64 1
  store i8 undef, i8* %a
  store i64 undef, i64* %z
  store i8 undef, i8* %a
  ; ^ future full elim phase should kill this
  store i8 4, i8* %P
  call void @may_throw()
  store i8 0, i8* %P
  store i8 undef, i8* %a
  br label %bb3
bb2:
  br label %bb3
bb3:
  ret i8* %P
}

define void @aliasing_load_kills(i8* %a) {
; CHECK-LABEL: @aliasing_load_kills(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 undef, label [[BB2:%.*]], label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 undef, i8* [[A:%.*]]
; CHECK-NEXT:    br label [[BB4:%.*]]
; CHECK:       bb3:
; CHECK-NEXT:    store i8 undef, i8* [[A]]
; CHECK-NEXT:    [[X:%.*]] = load i8, i8* [[A]]
; CHECK-NEXT:    store i8 undef, i8* [[A]]
; CHECK-NEXT:    br label [[BB4]]
; CHECK:       bb4:
; CHECK-NEXT:    ret void
;
bb0:
  store i8 undef, i8* %a
  br label %bb1
bb1:
  store i8 undef, i8* %a
  br i1 undef, label %bb2, label %bb3
bb2:
  store i8 undef, i8* %a
  br label %bb4
bb3:
  %x = load i8, i8* %a
  store i8 undef, i8* %a
  br label %bb4
bb4:
  ret void
}

define void @memcpy_example(i8* %a, i8* noalias %b, i1 %br0) {
; CHECK-LABEL: @memcpy_example(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[A:%.*]], i8* [[B:%.*]], i64 64, i32 8, i1 false)
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[A]], i8* [[B]], i64 64, i32 8, i1 false)
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    ret void
;
bb0:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 64, i32 8, i1 false)
  br i1 %br0, label %bb1, label %bb2
bb1:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 64, i32 8, i1 false)
  br label %bb3
bb2:
  br label %bb3
bb3:
  ret void
}

; http://i.imgur.com/abuFdZ2.png
define void @multiple_pre(i8* %a, i8 %b, i1 %c, i1 %d) {
; CHECK-LABEL: @multiple_pre(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[R:%.*]] = add i8 [[B:%.*]], 1
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[S:%.*]] = add i8 [[R]], 2
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    [[TMP0:%.*]] = phi i8* [ [[A:%.*]], [[BB2]] ], [ [[A]], [[BB1]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = phi i8 [ [[R]], [[BB2]] ], [ [[S]], [[BB1]] ]
; CHECK-NEXT:    br i1 [[D:%.*]], label [[BB4:%.*]], label [[BB5:%.*]]
; CHECK:       bb4:
; CHECK-NEXT:    store i8 [[R]], i8* [[A]]
; CHECK-NEXT:    br label [[EX:%.*]]
; CHECK:       bb5:
; CHECK-NEXT:    store i8 [[TMP1]], i8* [[TMP0]]
; CHECK-NEXT:    br label [[EX]]
; CHECK:       ex:
; CHECK-NEXT:    ret void
;
bb0:
  %r = add i8 %b, 1
  br i1 %c, label %bb1, label %bb2
bb1:
  %s = add i8 %r, 2
  store i8 %s, i8* %a
  br label %bb3
bb2:
  store i8 %r, i8* %a
  br label %bb3
bb3:
  br i1 %d, label %bb4, label %bb5
bb4:
  store i8 %r, i8* %a
  br label %ex
bb5:
  br label %ex
ex:
  ret void
}

; PRE insertion can't happen at the bb3 lambda because its uses, despite
; belonging to the same redundancy class, are altogether different instruction
; types. PDSE handles this by partitioning each redundancy class into
; like-opcode subclasses, and calculating anticipation separately for each.
;
; The i8* %a class below comprises of memset and store subclasses, and the bb3
; lambda counts as up-unsafe (and ultimately not willBeAnt) for both because:
; For the store subclass, bb3 is exposed to a memset (which counts as an
; aliasing, non-PRE-insertable occurrence), and vice versa for the memset
; subclass.
define void @unable_to_elim(i8* %a, i8 %b, i1 %c, i1 %d) {
; CHECK-LABEL: @unable_to_elim(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[R:%.*]] = add i8 [[B:%.*]], 1
; CHECK-NEXT:    br i1 [[C:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[S:%.*]] = add i8 [[R]], 2
; CHECK-NEXT:    call void @llvm.memset.p0i8.i64(i8* [[A:%.*]], i8 [[S]], i64 1, i32 1, i1 false)
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 [[R]], i8* [[A]]
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    br i1 [[D:%.*]], label [[BB4:%.*]], label [[BB5:%.*]]
; CHECK:       bb4:
; CHECK-NEXT:    store i8 [[R]], i8* [[A]]
; CHECK-NEXT:    br label [[EX:%.*]]
; CHECK:       bb5:
; CHECK-NEXT:    br label [[EX]]
; CHECK:       ex:
; CHECK-NEXT:    ret void
;
bb0:
  %r = add i8 %b, 1
  br i1 %c, label %bb1, label %bb2
bb1:
  %s = add i8 %r, 2
  call void @llvm.memset.p0i8.i64(i8* %a, i8 %s, i64 1, i32 1, i1 false)
  br label %bb3
bb2:
  store i8 %r, i8* %a
  br label %bb3
bb3:
  br i1 %d, label %bb4, label %bb5
bb4:
  store i8 %r, i8* %a
  br label %ex
bb5:
  br label %ex
ex:
  ret void
}

; FIXME: PDSE-ing
;
; a, b, c --- lambda ---
;               |
;               +------- c, b, a
;
; into:
;
; --- lambda --- a, b, c
;       |
;       +------- c, b, a
;
; would require multiple rounds of computeWillBeAnt.
define void @pre_blocked(i8* %a, i8* %b, i8* %c, i1 %br0) {
; CHECK-LABEL: @pre_blocked(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    store i8 1, i8* [[A:%.*]]
; CHECK-NEXT:    store i8 1, i8* [[B:%.*]]
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    store i8 1, i8* [[C:%.*]]
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 11, i8* [[C]]
; CHECK-NEXT:    store i8 11, i8* [[B]]
; CHECK-NEXT:    store i8 11, i8* [[A]]
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    ret void
;
bb0:
  store i8 1, i8* %a
  store i8 1, i8* %b
  store i8 1, i8* %c
  br i1 %br0, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  store i8 11, i8* %c
  store i8 11, i8* %b
  store i8 11, i8* %a
  br label %bb3
bb3:
  ret void
}

; FIXME: Should transform this:
;
; s, s' --- lambda ---
;             |
;             +------- s
;
; into:
;
; --- lambda --- s, s'
;       |
;       +------- s', s
define void @pre_blocked_again(i64* %a, i1 %br0) {
; CHECK-LABEL: @pre_blocked_again(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    store i64 1, i64* [[A:%.*]]
; CHECK-NEXT:    [[X:%.*]] = bitcast i64* [[A]] to i8*
; CHECK-NEXT:    [[B:%.*]] = getelementptr i8, i8* [[X]], i64 1
; CHECK-NEXT:    store i8 2, i8* [[B]]
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    store i64 1, i64* [[A]]
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    ret void
;
bb0:
  store i64 1, i64* %a
  %x = bitcast i64* %a to i8*
  %b = getelementptr i8, i8* %x, i64 1
  store i8 2, i8* %b
  br i1 %br0, label %bb1, label %bb2
bb1:
  store i64 1, i64* %a
  br label %bb3
bb2:
  br label %bb3
bb3:
  ret void
}

define void @never_escapes() {
; CHECK-LABEL: @never_escapes(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    ret void
;
bb0:
  %a = alloca i8
  br label %bb1
bb1:
  store i8 12, i8* %a
  br label %bb2
bb2:
  ret void
}

define void @propagate_up_unsafety(i8* %a, i1 %br0, i1 %br1, i1 %br2) {
; CHECK-LABEL: @propagate_up_unsafety(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[BR2:%.*]], label [[BB5:%.*]], label [[BB6:%.*]]
; CHECK:       bb5:
; CHECK-NEXT:    br label [[BB0:%.*]]
; CHECK:       bb6:
; CHECK-NEXT:    br label [[BB0]]
; CHECK:       bb0:
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 [[BR1:%.*]], label [[BB3:%.*]], label [[BB4:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 1, i8* [[A:%.*]]
; CHECK-NEXT:    ret void
; CHECK:       bb3:
; CHECK-NEXT:    br label [[BB0]]
; CHECK:       bb4:
; CHECK-NEXT:    store i8 3, i8* [[A]]
; CHECK-NEXT:    ret void
;
entry:
  br i1 %br2, label %bb5, label %bb6
bb5:
  br label %bb0
bb6:
  store i8 12, i8* %a
  br label %bb0
bb0:
  br i1 %br0, label %bb1, label %bb2
bb1:
  br i1 %br1, label %bb3, label %bb4
bb2:
  store i8 1, i8* %a
  ret void
bb3:
  br label %bb0
bb4:
  store i8 3, i8* %a
  ret void
}

define void @multiple_insertions(i8* %a, i32 %br0) {
; CHECK-LABEL: @multiple_insertions(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    switch i32 [[BR0:%.*]], label [[BB1:%.*]] [
; CHECK-NEXT:    i32 1, label [[BB2:%.*]]
; CHECK-NEXT:    i32 2, label [[BB3:%.*]]
; CHECK-NEXT:    ]
; CHECK:       bb1:
; CHECK-NEXT:    store i8 12, i8* [[A:%.*]]
; CHECK-NEXT:    br label [[BB4:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 12, i8* [[A]]
; CHECK-NEXT:    br label [[BB4]]
; CHECK:       bb3:
; CHECK-NEXT:    store i8 12, i8* [[A]]
; CHECK-NEXT:    br label [[BB4]]
; CHECK:       bb4:
; CHECK-NEXT:    ret void
;
bb0:
  store i8 12, i8* %a
  switch i32 %br0, label %bb1 [
  i32 1, label %bb2
  i32 2, label %bb3
  ]
bb1:
  store i8 12, i8* %a
  br label %bb4
bb2:
  br label %bb4
bb3:
  br label %bb4
bb4:
  ret void
}

; The lambda at bb1 can be PRE-ed, but the insertion comes from behind an
; earlier lambda -- the one at bb0.
define void @propagate_uses(i8* %a, i1 %br0, i1 %br1) {
; CHECK-LABEL: @propagate_uses(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 [[BR0]], label [[BB3:%.*]], label [[BB1_BB4_CRIT_EDGE:%.*]]
; CHECK:       bb1.bb4_crit_edge:
; CHECK-NEXT:    store i8 1, i8* [[A:%.*]]
; CHECK-NEXT:    br label [[BB4:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 1, i8* [[A]]
; CHECK-NEXT:    br label [[BB5:%.*]]
; CHECK:       bb3:
; CHECK-NEXT:    store i8 2, i8* [[A]]
; CHECK-NEXT:    br label [[BB4]]
; CHECK:       bb4:
; CHECK-NEXT:    br label [[BB5]]
; CHECK:       bb5:
; CHECK-NEXT:    ret void
;
bb0:
  store i8 1, i8* %a
  br i1 %br0, label %bb1, label %bb2
bb1:
  br i1 %br0, label %bb3, label %bb4
bb2:
  br label %bb5
bb3:
  store i8 2, i8* %a
  br label %bb4
bb4:
  br label %bb5
bb5:
  ret void
}

; Check that renaming rules handle overwrites correctly.
define void @small_store_can_dse(i8*, i8* noalias) {
; CHECK-LABEL: @small_store_can_dse(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i8, i8* [[TMP0:%.*]], i64 2
; CHECK-NEXT:    [[TMP3:%.*]] = load i8, i8* [[TMP2]]
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i64*
; CHECK-NEXT:    store i64 3, i64* [[TMP4]]
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP5:%.*]] = load i8, i8* [[TMP0]]
; CHECK-NEXT:    [[TMP6:%.*]] = load i8, i8* [[TMP2]]
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP0]], i8* [[TMP1:%.*]], i64 8, i32 1, i1 false)
; CHECK-NEXT:    ret void
;
bb0:
  store i8 1, i8* %0
  %2 = getelementptr inbounds i8, i8* %0, i64 2
  %3 = load i8, i8* %2
  %4 = bitcast i8* %0 to i64*
  store i64 3, i64* %4
  br label %bb1
bb1:
  %5 = load i8, i8* %0
  store i8 1, i8* %0
  %6 = load i8, i8* %2
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 8, i32 1, i1 false)
  ret void
}

; Nothing should be inserted. The i64 store should be used by the lambda in bb0.
define void @no_extra_insert(i8* %a, i1 %br0) {
; CHECK-LABEL: @no_extra_insert(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[B:%.*]] = bitcast i8* [[A:%.*]] to i64*
; CHECK-NEXT:    store i64 2, i64* [[B]]
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 3, i8* [[A]]
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
bb0:
  store i8 1, i8* %a
  br i1 %br0, label %bb1, label %bb2
bb1:
  %b = bitcast i8* %a to i64*
  store i64 2, i64* %b
  br label %exit
bb2:
  store i8 3, i8* %a
  br label %exit
exit:
  ret void
}

; No DSE because the memmove doubles as an aliasing load.
define void @kills_own_occ_class(i8* %a, i8* %b) {
; CHECK-LABEL: @kills_own_occ_class(
; CHECK-NEXT:    store i8 3, i8* [[A:%.*]]
; CHECK-NEXT:    call void @llvm.memmove.p0i8.p0i8.i64(i8* [[A]], i8* nonnull [[B:%.*]], i64 1, i32 8, i1 false)
; CHECK-NEXT:    ret void
;
  store i8 3, i8* %a
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %a, i8* nonnull %b, i64 1, i32 8, i1 false)
  ret void
}

; i1* %tmp and i8* %arg belong to the same redundancy class, but have different
; types. So subclasses need to be grouped by the store value operand type in
; addition to opcode.
define void @subclass_on_store_value_type_too(i8* %arg) {
; CHECK-LABEL: @subclass_on_store_value_type_too(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP:%.*]] = bitcast i8* [[ARG:%.*]] to i1*
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 undef, label [[BB1]], label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i1 undef, i1* [[TMP]]
; CHECK-NEXT:    ret void
;
bb:
  store i8 -123, i8* %arg
  %tmp = bitcast i8* %arg to i1*
  br label %bb1
bb1:
  store i1 undef, i1* %tmp
  br i1 undef, label %bb1, label %bb2
bb2:
  ret void
}

define void @dont_include_nonsubclass_uses(i8* %arg) {
; CHECK-LABEL: @dont_include_nonsubclass_uses(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 undef, label [[BB1]], label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 -123, i8* [[ARG:%.*]]
; CHECK-NEXT:    ret void
;
bb:
  %tmp = bitcast i8* %arg to i1*
  br label %bb1
bb1:
  store i1 undef, i1* %tmp
  br i1 undef, label %bb1, label %bb2
bb2:
  store i8 -123, i8* %arg
  ret void
}

; The critical edge from the lambda at bb4 to bb6 is an unsplittable, which
; prevents PRE from filling its corresponding null def. PDSE therefore
; classifies bb4 as `CanBeAnt == false`.
define void @cant_split_indirectbr_edge() {
; CHECK-LABEL: @cant_split_indirectbr_edge(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB2:%.*]], label %bb6]
; CHECK:       bb2:
; CHECK-NEXT:    indirectbr i8* undef, [label %bb3]
; CHECK:       bb3:
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB4:%.*]], label %bb3]
; CHECK:       bb4:
; CHECK-NEXT:    store float undef, float* undef, align 1
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB2]], label %bb6]
; CHECK:       bb6:
; CHECK-NEXT:    ret void
;
bb:
  indirectbr i8* undef, [label %bb2, label %bb6]
bb2:
  indirectbr i8* undef, [label %bb3]
bb3:
  store float undef, float* undef, align 1
  indirectbr i8* undef, [label %bb4, label %bb3]
bb4:
  indirectbr i8* undef, [label %bb2, label %bb6]
bb6:
  ret void
}

; Same as cant_split_indirectbr_edge, but with a lambda use.
define void @cant_split_indirectbr_edge2() {
; CHECK-LABEL: @cant_split_indirectbr_edge2(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB2:%.*]], label [[BB4:%.*]], label %bb6]
; CHECK:       bb2:
; CHECK-NEXT:    indirectbr i8* undef, [label %bb3]
; CHECK:       bb3:
; CHECK-NEXT:    store float undef, float* undef, align 1
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB4]], label %bb3]
; CHECK:       bb4:
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB2]], label %bb6]
; CHECK:       bb6:
; CHECK-NEXT:    ret void
;
bb:
  indirectbr i8* undef, [label %bb2, label %bb4, label %bb6]
bb2:
  indirectbr i8* undef, [label %bb3]
bb3:
  store float undef, float* undef, align 1
  indirectbr i8* undef, [label %bb4, label %bb3]
bb4:
  indirectbr i8* undef, [label %bb2, label %bb6]
bb6:
  ret void
}

define void @cant_split2(i8* %arg) {
; CHECK-LABEL: @cant_split2(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB1:%.*]], label %bb2]
; CHECK:       bb1:
; CHECK-NEXT:    store i8* blockaddress(@cant_split2, [[BB2:%.*]]), i8** undef, align 4
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB5:%.*]], label %bb1]
; CHECK:       bb2:
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB4:%.*]], label %bb3]
; CHECK:       bb3:
; CHECK-NEXT:    indirectbr i8* undef, [label [[BB1]], label %bb5]
; CHECK:       bb4:
; CHECK-NEXT:    unreachable
; CHECK:       bb5:
; CHECK-NEXT:    ret void
;
bb:
  indirectbr i8* undef, [label %bb1, label %bb2]
bb1:
  store i8* blockaddress(@cant_split2, %bb2), i8** undef, align 4
  indirectbr i8* undef, [label %bb5, label %bb1]
bb2:
  indirectbr i8* undef, [label %bb4, label %bb3]
bb3:
  indirectbr i8* undef, [label %bb1, label %bb5]
bb4:
  unreachable
bb5:
  ret void
}

; A store is loop-variant iff its pointer's SSA def graph contains an SCC. Such
; an SCC necessarily contains a phi, so reset the top of stack to _|_ upon
; exiting the phi's block.
define void @dont_sink_loop_variant_store(i8* %a, i8 %len) {
; CHECK-LABEL: @dont_sink_loop_variant_store(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[DONE:%.*]] = icmp eq i8 [[LEN:%.*]], 0
; CHECK-NEXT:    br i1 [[DONE]], label [[BB3:%.*]], label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[COUNT:%.*]] = phi i8 [ [[LEN]], [[BB0:%.*]] ], [ [[NEXT:%.*]], [[BB2:%.*]] ]
; CHECK-NEXT:    [[NEXT]] = sub i8 [[COUNT]], 1
; CHECK-NEXT:    [[LOC:%.*]] = getelementptr i8, i8* [[A:%.*]], i8 [[NEXT]]
; CHECK-NEXT:    br label [[BB2]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 [[NEXT]], i8* [[LOC]]
; CHECK-NEXT:    [[DONEYET:%.*]] = icmp eq i8 [[NEXT]], 0
; CHECK-NEXT:    br i1 [[DONEYET]], label [[BB3]], label [[BB1]]
; CHECK:       bb3:
; CHECK-NEXT:    ret void
;
bb0:
  %done = icmp eq i8 %len, 0
  br i1 %done, label %bb3, label %bb1
bb1:
  %count = phi i8 [ %len, %bb0 ], [ %next, %bb2 ]
  %next = sub i8 %count, 1
  %loc = getelementptr i8, i8* %a, i8 %next
  br label %bb2
bb2:
  store i8 %next, i8* %loc      ; Loop-variant store.
  %doneyet = icmp eq i8 %next, 0
  br i1 %doneyet, label %bb3, label %bb1
bb3:
  ret void
}

; Same as above, but with a loop-invariant store.
define void @sink_loop_invariant_store(i8* %a, i8 %len) {
; CHECK-LABEL: @sink_loop_invariant_store(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[DONE:%.*]] = icmp eq i8 [[LEN:%.*]], 0
; CHECK-NEXT:    br i1 [[DONE]], label [[BB3:%.*]], label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[COUNT:%.*]] = phi i8 [ [[LEN]], [[BB0:%.*]] ], [ [[NEXT:%.*]], [[BB2:%.*]] ]
; CHECK-NEXT:    [[NEXT]] = sub i8 [[COUNT]], 1
; CHECK-NEXT:    [[LOC:%.*]] = getelementptr i8, i8* [[A:%.*]], i8 [[NEXT]]
; CHECK-NEXT:    br label [[BB2]]
; CHECK:       bb2:
; CHECK-NEXT:    [[DONEYET:%.*]] = icmp eq i8 [[NEXT]], 0
; CHECK-NEXT:    br i1 [[DONEYET]], label [[BB2_BB3_CRIT_EDGE:%.*]], label [[BB1]]
; CHECK:       bb2.bb3_crit_edge:
; CHECK-NEXT:    store i8 [[NEXT]], i8* [[A]]
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    ret void
;
bb0:
  %done = icmp eq i8 %len, 0
  br i1 %done, label %bb3, label %bb1
bb1:
  %count = phi i8 [ %len, %bb0 ], [ %next, %bb2 ]
  %next = sub i8 %count, 1
  %loc = getelementptr i8, i8* %a, i8 %next
  br label %bb2
bb2:
  store i8 %next, i8* %a
  %doneyet = icmp eq i8 %next, 0
  br i1 %doneyet, label %bb3, label %bb1
bb3:
  ret void
}

; Loop-variant store that connects to more than one SCC. Should not be sunk.
define void @multiple_scc_phis(i8* %a) {
; CHECK-LABEL: @multiple_scc_phis(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[P0:%.*]] = phi i8 [ 0, [[BB0:%.*]] ], [ [[NEXTP0:%.*]], [[BB1]] ]
; CHECK-NEXT:    [[NEXTP0]] = add i8 1, [[P0]]
; CHECK-NEXT:    br i1 undef, label [[BB1]], label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    [[P1:%.*]] = phi i8 [ [[P0]], [[BB1]] ], [ [[NEXTP1:%.*]], [[BB2]] ]
; CHECK-NEXT:    [[NEXTP1]] = add i8 1, [[P1]]
; CHECK-NEXT:    [[B:%.*]] = getelementptr i8, i8* [[A:%.*]], i8 [[NEXTP1]]
; CHECK-NEXT:    store i8 [[NEXTP1]], i8* [[B]]
; CHECK-NEXT:    br i1 undef, label [[BB2]], label [[BB3:%.*]]
; CHECK:       bb3:
; CHECK-NEXT:    ret void
;
bb0:
  br label %bb1
bb1:
  %p0 = phi i8 [0, %bb0], [%nextp0, %bb1]
  %nextp0 = add i8 1, %p0
  br i1 undef, label %bb1, label %bb2
bb2:
  %p1 = phi i8 [%p0, %bb1], [%nextp1, %bb2]
  %nextp1 = add i8 1, %p1
  %b = getelementptr i8, i8* %a, i8 %nextp1
  store i8 %nextp1, i8* %b
  br i1 undef, label %bb2, label %bb3
bb3:
  ret void
}

; Not really loop-variant, even though %offset's def graph has an SCC.
define void @phidef(i8* %a, i1 %br0, i1 %br1) {
; CHECK-LABEL: @phidef(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    [[OFFSET:%.*]] = phi i8 [ 123, [[BB1]] ], [ 65, [[BB2]] ], [ [[OFFSET]], [[BB4:%.*]] ]
; CHECK-NEXT:    [[LOC:%.*]] = getelementptr i8, i8* [[A:%.*]], i8 [[OFFSET]]
; CHECK-NEXT:    br label [[BB4]]
; CHECK:       bb4:
; CHECK-NEXT:    br i1 [[BR1:%.*]], label [[BB5:%.*]], label [[BB3]]
; CHECK:       bb5:
; CHECK-NEXT:    store i8 [[OFFSET]], i8* [[LOC]]
; CHECK-NEXT:    ret void
;
bb0:
  br i1 %br0, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  br label %bb3
bb3:
  %offset = phi i8 [ 123, %bb1 ], [ 321, %bb2 ], [ %offset, %bb4 ]
  %loc = getelementptr i8, i8* %a, i8 %offset
  br label %bb4
bb4:
  store i8 %offset, i8* %loc
  br i1 %br1, label %bb5, label %bb3
bb5:
  ret void
}

; %loc -> phi(bb3) -> phi(bb2) <-> %s
;                    ^               ^
;                    +----- SCC -----+
; No sink, even though the store exists outside of the inner loop.
define void @defined_by_inductive(i8* %a) {
; CHECK-LABEL: @defined_by_inductive(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 undef, label [[BB2:%.*]], label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    [[R:%.*]] = phi i8 [ 41, [[BB1]] ], [ [[S:%.*]], [[BB2]] ]
; CHECK-NEXT:    [[S]] = add i8 [[R]], 1
; CHECK-NEXT:    br i1 undef, label [[BB2]], label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    [[OFFSET:%.*]] = phi i8 [ 0, [[BB1]] ], [ [[R]], [[BB2]] ]
; CHECK-NEXT:    [[LOC:%.*]] = getelementptr i8, i8* [[A:%.*]], i8 [[OFFSET]]
; CHECK-NEXT:    br i1 undef, label [[BB1]], label [[BB4:%.*]]
; CHECK:       bb4:
; CHECK-NEXT:    store i8 [[OFFSET]], i8* [[LOC]]
; CHECK-NEXT:    ret void
;
bb0:
  br label %bb1
bb1:
  br i1 undef, label %bb2, label %bb3
bb2:
  %r = phi i8 [41, %bb1] , [%s, %bb2]
  %s = add i8 %r, 1
  br i1 undef, label %bb2, label %bb3
bb3:
  %offset = phi i8 [ 1024, %bb1 ], [ %r, %bb2 ]
  %loc = getelementptr i8, i8* %a, i8 %offset
  store i8 %offset, i8* %loc
  br i1 undef, label %bb1, label %bb4
bb4:
  ret void
}

; Calling `free` equivalent to a DeadOnExit occurrence. TODO: Treat it instead
; like a volatile real occurrence that `!canDSE()`.
define void @test_free(i1 %br0) {
; CHECK-LABEL: @test_free(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[X:%.*]] = call i8* @malloc(i32 4)
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    call void @free(i8* [[X]])
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 1, i8* [[X]]
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    [[USE:%.*]] = load i8, i8* [[X]]
; CHECK-NEXT:    ret void
;
bb0:
  %x = call i8* @malloc(i32 4)
  store i8 1, i8* %x
  br i1 %br0, label %bb1, label %bb2
bb1:
  call void @free(i8* %x)
  br label %bb3
bb2:
  br label %bb3
bb3:
  %use = load i8, i8* %x
  ret void
}

; Avoid PRE insertions into catchswitch blocks. In the future, consider
; deferring insertion into each catchpad block.
define void @catchswitch_noninsertable() personality i32 (...)* @personality {
; CHECK-LABEL: @catchswitch_noninsertable(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP:%.*]] = alloca i8
; CHECK-NEXT:    store i8 13, i8* [[TMP]]
; CHECK-NEXT:    invoke void @may_throw()
; CHECK-NEXT:    to label [[BB5:%.*]] unwind label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP2:%.*]] = catchswitch within none [label %bb3] unwind to caller
; CHECK:       bb3:
; CHECK-NEXT:    [[TMP4:%.*]] = catchpad within [[TMP2]] [i8* null, i32 64, i8* null]
; CHECK-NEXT:    unreachable
; CHECK:       bb5:
; CHECK-NEXT:    unreachable
;
bb:
  %tmp = alloca i8
  store i8 13, i8* %tmp
  invoke void @may_throw()
  to label %bb5 unwind label %bb1

bb1:
  %tmp2 = catchswitch within none [label %bb3] unwind to caller

bb3:
  %tmp4 = catchpad within %tmp2 [i8* null, i32 64, i8* null]
  unreachable

bb5:
  unreachable
}

; Like the above, but we can PRE into LandingPadInst blocks.
define void @landingpad_insertable() personality i32 (...)* @personality {
; CHECK-LABEL: @landingpad_insertable(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP:%.*]] = call i8* @malloc(i32 4)
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i8* [[TMP]] to i32*
; CHECK-NEXT:    invoke void @may_throw()
; CHECK-NEXT:    to label [[BB2:%.*]] unwind label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    unreachable
; CHECK:       bb3:
; CHECK-NEXT:    [[TMP4:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:    catch i8* null
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i32* [[TMP1]] to i8*
; CHECK-NEXT:    call void @free(i8* [[TMP5]])
; CHECK-NEXT:    unreachable
;
bb:
  %tmp = call i8* @malloc(i32 4)
  %tmp1 = bitcast i8* %tmp to i32*
  store i32 undef, i32* %tmp1, align 4
  invoke void @may_throw()
  to label %bb2 unwind label %bb3

bb2:
  unreachable

bb3:
  %tmp4 = landingpad { i8*, i32 }
  catch i8* null
  %tmp5 = bitcast i32* %tmp1 to i8*
  call void @free(i8* %tmp5)
  unreachable
}

define void @ssaupdater_crash() {
; CHECK-LABEL: @ssaupdater_crash(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP:%.*]] = alloca i64, align 8
; CHECK-NEXT:    br label [[BB5:%.*]]
; CHECK:       bb5:
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i64* [[TMP]] to i8*
; CHECK-NEXT:    call void @llvm.memset.p0i8.i64(i8* nonnull [[TMP6]], i8 0, i64 16, i32 8, i1 false)
; CHECK-NEXT:    br i1 undef, label [[BB7:%.*]], label [[BB8:%.*]]
; CHECK:       bb7:
; CHECK-NEXT:    br i1 undef, label [[BB9:%.*]], label [[BB8]]
; CHECK:       bb8:
; CHECK-NEXT:    unreachable
; CHECK:       bb9:
; CHECK-NEXT:    [[TMP10:%.*]] = call i64 @f(i64* [[TMP]])
; CHECK-NEXT:    unreachable
;
bb:
  %tmp = alloca i64, align 8
  br label %bb5

bb5:
  %tmp6 = bitcast i64* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %tmp6, i8 0, i64 16, i32 8, i1 false)
  br i1 undef, label %bb7, label %bb8

bb7:
  br i1 undef, label %bb9, label %bb8

bb8:
  unreachable

bb9:
  %tmp10 = call i64 @f(i64* %tmp)
  unreachable
}

; Ensure that returned memory isn't counted as dead on exit. Since %x is
; returned, the store in bb1 should remain.
define i8* @not_dead_on_exit(i1 %br0) {
; CHECK-LABEL: @not_dead_on_exit(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[X:%.*]] = call i8* @malloc(i32 1)
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    store i8 23, i8* [[X]]
; CHECK-NEXT:    br label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[BB3]]
; CHECK:       bb3:
; CHECK-NEXT:    [[RV:%.*]] = phi i8* [ [[X]], [[BB1]] ], [ null, [[BB2]] ]
; CHECK-NEXT:    ret i8* [[RV]]
;
bb0:
  %x = call i8* @malloc(i32 1)
  br i1 %br0, label %bb1, label %bb2
bb1:
  store i8 23, i8* %x
  br label %bb3
bb2:
  br label %bb3
bb3:
  %rv = phi i8* [%x, %bb1], [null, %bb2]
  ret i8* %rv
}

; There are identical critical edges from bb0 to bb2 that should merge into the
; same predecessor of the split edge.
define void @multiple_edges_same_successor(i8* %a, i32 %b) {
; CHECK-LABEL: @multiple_edges_same_successor(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    switch i32 [[B:%.*]], label [[BB1:%.*]] [
; CHECK-NEXT:    i32 0, label [[BB0_BB2_CRIT_EDGE:%.*]]
; CHECK-NEXT:    i32 1, label [[BB0_BB2_CRIT_EDGE]]
; CHECK-NEXT:    ]
; CHECK:       bb0.bb2_crit_edge:
; CHECK-NEXT:    store i8 1, i8* [[A:%.*]]
; CHECK-NEXT:    br label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    store i8 2, i8* [[A]]
; CHECK-NEXT:    br label [[BB2]]
; CHECK:       bb2:
; CHECK-NEXT:    ret void
;
bb0:
  store i8 1, i8* %a
  switch i32 %b, label %bb1 [
  i32 0, label %bb2
  i32 1, label %bb2
  ]
bb1:
  store i8 2, i8* %a
  br label %bb2
bb2:
  ret void
}

; Writes to memory of unknown size should not eliminate or be eliminated.
define i8* @unknown_memory_location_size(i32 %len, i8* %a) {
; CHECK-LABEL: @unknown_memory_location_size(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[X:%.*]] = call i8* @malloc(i32 [[LEN:%.*]])
; CHECK-NEXT:    [[SETLEN:%.*]] = zext i32 [[LEN]] to i64
; CHECK-NEXT:    call void @llvm.memset.p0i8.i64(i8* [[X]], i8 0, i64 [[SETLEN]], i32 1, i1 false)
; CHECK-NEXT:    [[CPYLEN:%.*]] = and i64 [[SETLEN]], 65536
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[X]], i8* [[A:%.*]], i64 [[CPYLEN]], i32 1, i1 false)
; CHECK-NEXT:    ret i8* [[X]]
;
bb0:
  %x = call i8* @malloc(i32 %len)
  %setlen = zext i32 %len to i64
  call void @llvm.memset.p0i8.i64(i8* %x, i8 0, i64 %setlen, i32 1, i1 false)
  %cpylen = and i64 %setlen, 65536
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %x, i8* %a, i64 %cpylen, i32 1, i1 false)
  ret i8* %x
}

; TODO: Treat lifetime_end like a volatile real occurrence -- it can eliminate
; any exposed must-aliasing store post-dommed by it, but should not be subject
; to elimination or PRE insertion.
define void @lifetime_end_test(i8* %a, i1 %br0) {
; CHECK-LABEL: @lifetime_end_test(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    call void @llvm.lifetime.start(i64 1, i8* nonnull [[A:%.*]])
; CHECK-NEXT:    store i8 23, i8* [[A]]
; CHECK-NEXT:    call void @llvm.lifetime.end(i64 1, i8* nonnull [[A]])
; CHECK-NEXT:    call void @llvm.lifetime.start(i64 1, i8* nonnull [[A]])
; CHECK-NEXT:    store i8 29, i8* [[A]]
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    call void @llvm.lifetime.end(i64 1, i8* nonnull [[A]])
; CHECK-NEXT:    ret void
; CHECK:       bb2:
; CHECK-NEXT:    [[TMP:%.*]] = load i8, i8* [[A]]
; CHECK-NEXT:    call void @llvm.lifetime.end(i64 1, i8* nonnull [[A]])
; CHECK-NEXT:    ret void
;
bb0:
  call void @llvm.lifetime.start(i64 1, i8* nonnull %a)
  store i8 23, i8* %a
  call void @llvm.lifetime.end(i64 1, i8* nonnull %a)
  call void @llvm.lifetime.start(i64 1, i8* nonnull %a)
  store i8 29, i8* %a
  br i1 %br0, label %bb1, label %bb2
bb1:
  call void @llvm.lifetime.end(i64 1, i8* nonnull %a)
  ret void
bb2:
  %tmp = load i8, i8* %a
  call void @llvm.lifetime.end(i64 1, i8* nonnull %a)
  ret void
}
