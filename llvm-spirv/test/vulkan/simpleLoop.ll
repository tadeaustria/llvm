; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv --vulkan --spirv-ext=+all %t.bc -o %t.spv
; RUNx: llvm-spirv -to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val --target-env vulkan1.1 %t.spv

; ModuleID = 'mainVO3.bc'
source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-vulkan-windows-sycldevice"

%struct._ZTS8_arg_0_t._arg_0_t = type { i32 addrspace(10)* }
%struct._ZTS8_arg_2_t._arg_2_t = type { %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" }
%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" = type { [2 x i64] }
%struct._ZTS8_arg_3_t._arg_3_t = type { %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" }
%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE6VecAdd" = comdat any

@_arg_0 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_0_t._arg_0_t, align 8
@_arg_2 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_2_t._arg_2_t, align 8
@_arg_3 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_3_t._arg_3_t, align 8
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(7) constant <3 x i32>, align 16

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE6VecAdd"() local_unnamed_addr #0 comdat !kernel_arg_addr_space !5 !kernel_arg_access_qual !5 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !5 !kernel_arg_buffer_location !5 {
entry:
  %0 = load i32 addrspace(10)*, i32 addrspace(10)* addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_0_t._arg_0_t, %struct._ZTS8_arg_0_t._arg_0_t addrspace(10)* @_arg_0, i64 0, i32 0), align 8, !tbaa !6
  %agg.tmp1.sroa.0.sroa.2.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_2_t._arg_2_t, %struct._ZTS8_arg_2_t._arg_2_t addrspace(10)* @_arg_2, i64 0, i32 0, i32 0, i32 0, i64 1), align 8
  %1 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInGlobalInvocationId, align 16, !noalias !11
  %2 = extractelement <3 x i32> %1, i64 1
  %conv.i.i.i.i.i = zext i32 %2 to i64
  %cmp35.not.i = icmp eq i32 %2, 0
  br i1 %cmp35.not.i, label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILi2EEEE_clES5_.exit", label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %entry
  %3 = extractelement <3 x i32> %1, i64 0
  %conv.i.i2.i.i.i = zext i32 %3 to i64
  %agg.tmp2.sroa.0.sroa.2.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_3_t._arg_3_t, %struct._ZTS8_arg_3_t._arg_3_t addrspace(10)* @_arg_3, i64 0, i32 0, i32 0, i32 0, i64 1), align 8
  %agg.tmp2.sroa.0.sroa.0.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_3_t._arg_3_t, %struct._ZTS8_arg_3_t._arg_3_t addrspace(10)* @_arg_3, i64 0, i32 0, i32 0, i32 0, i64 0), align 8
  %add6.i.i.i = add i64 %agg.tmp2.sroa.0.sroa.0.0.copyload, %conv.i.i.i.i.i
  %mul.1.i.i.i = mul i64 %add6.i.i.i, %agg.tmp1.sroa.0.sroa.2.0.copyload
  %add.1.i.i.i = add i64 %agg.tmp2.sroa.0.sroa.2.0.copyload, %conv.i.i2.i.i.i
  %add6.1.i.i.i = add i64 %mul.1.i.i.i, %add.1.i.i.i
  %ptridx.i.i = getelementptr inbounds i32, i32 addrspace(10)* %0, i64 %add6.1.i.i.i
  %ptridx.ascast.i.i = addrspacecast i32 addrspace(10)* %ptridx.i.i to i32*
  %.pre = load i32, i32* %ptridx.ascast.i.i, align 4, !tbaa !18
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
  %4 = phi i32 [ %.pre, %for.body.lr.ph.i ], [ %add.i, %for.body.i ]
  %i.036.i = phi i64 [ 0, %for.body.lr.ph.i ], [ %inc.i, %for.body.i ]
  %add6.i.i.i.i = sub i64 %add6.i.i.i, %i.036.i
  %mul.1.i.i.i.i = mul i64 %add6.i.i.i.i, %agg.tmp1.sroa.0.sroa.2.0.copyload
  %add6.1.i.i.i.i = add i64 %mul.1.i.i.i.i, %add.1.i.i.i
  %ptridx.i.i.i = getelementptr inbounds i32, i32 addrspace(10)* %0, i64 %add6.1.i.i.i.i
  %ptridx.ascast.i.i.i = addrspacecast i32 addrspace(10)* %ptridx.i.i.i to i32*
  %5 = load i32, i32* %ptridx.ascast.i.i.i, align 4, !tbaa !18
  %add.i = add nsw i32 %5, %4
  store i32 %add.i, i32* %ptridx.ascast.i.i, align 4, !tbaa !18
  %inc.i = add nuw nsw i64 %i.036.i, 1
  %exitcond.not = icmp eq i64 %inc.i, %conv.i.i.i.i.i
  br i1 %exitcond.not, label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILi2EEEE_clES5_.exit.loopexit", label %for.body.i

"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILi2EEEE_clES5_.exit.loopexit": ; preds = %for.body.i
  br label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILi2EEEE_clES5_.exit"

"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILi2EEEE_clES5_.exit": ; preds = %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_2idILi2EEEE_clES5_.exit.loopexit", %entry
  ret void
}

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="main.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dependent-libraries = !{!0}
!llvm.module.flags = !{!1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 12.0.0 (git@github.com:tadeaustria/llvm.git 010777c72dc5fb7c43be2cb7658f8762643e40a5)"}
!5 = !{}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS8_arg_0_t", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !14, !16}
!12 = distinct !{!12, !13, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv: %agg.result"}
!13 = distinct !{!13, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv"}
!14 = distinct !{!14, !15, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v: %agg.result"}
!15 = distinct !{!15, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v"}
!16 = distinct !{!16, !17, !"_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_2idIXT_EEEPS5_: %agg.result"}
!17 = distinct !{!17, !"_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_2idIXT_EEEPS5_"}
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !9, i64 0}