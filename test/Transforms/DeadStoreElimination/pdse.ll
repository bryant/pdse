; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -pdse %s | FileCheck %s

declare void @may_throw()
declare noalias i8* @malloc(i32)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64 , i32 , i1 )
declare void @llvm.memmove.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i32, i1)

define void @lo_and_chow(i8* %x, i1 %br0, i1 %br1) {
; CHECK-LABEL: @lo_and_chow(
; CHECK-NEXT:  bb0:
; CHECK-NEXT:    [[V:%.*]] = load i8, i8* [[X:%.*]]
; CHECK-NEXT:    [[V1:%.*]] = add nuw i8 [[V]], 1
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br i1 [[BR0:%.*]], label [[BB2:%.*]], label [[BB3:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    store i8 [[V1]], i8* [[X]]
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

define void @lo_and_chow_maythrow(i8* %x, i1 %br0, i1 %br1) {
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

define void @memcpy_example(i8* %a, i8* %b, i1 %br0) {
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
; CHECK-NEXT:    [[TMP0:%.*]] = phi i8 [ [[R]], [[BB2]] ], [ [[S]], [[BB1]] ]
; CHECK-NEXT:    br i1 [[D:%.*]], label [[BB4:%.*]], label [[BB5:%.*]]
; CHECK:       bb4:
; CHECK-NEXT:    store i8 [[R]], i8* [[A:%.*]]
; CHECK-NEXT:    br label [[EX:%.*]]
; CHECK:       bb5:
; CHECK-NEXT:    store i8 [[TMP0]], i8* [[A]]
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
; would require multiple rounds of willBeAnt
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
;       +------- s'
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

; Check that renaming rules handle overwrites correctly.
define void @small_store_can_dse(i8*, i8* noalias) {
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

; Nothing should be inserted. The i64 store should be used by the i8 lambda in
; bb1.
define void @no_extra_insert(i8* %a, i1 %br0) {
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
