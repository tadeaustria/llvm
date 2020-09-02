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
%struct._ZTS8_arg_4_t._arg_4_t = type { i32 addrspace(10)* }
%struct._ZTS8_arg_6_t._arg_6_t = type { %"class._ZTSN2cl4sycl5rangeILi3EEE.cl::sycl::range" }
%"class._ZTSN2cl4sycl5rangeILi3EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi3EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi3EEE.cl::sycl::detail::array" = type { [3 x i64] }
%struct._ZTS8_arg_7_t._arg_7_t = type { %"class._ZTSN2cl4sycl2idILi3EEE.cl::sycl::id" }
%"class._ZTSN2cl4sycl2idILi3EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi3EEE.cl::sycl::detail::array" }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11dim2_subscr" = comdat any

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11dim3_subscr" = comdat any

@_arg_0 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_0_t._arg_0_t, align 8
@_arg_2 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_2_t._arg_2_t, align 8
@_arg_3 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_3_t._arg_3_t, align 8
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(7) constant <3 x i32>, align 16
@__spirv_BuiltInNumWorkgroups = external dso_local local_unnamed_addr addrspace(7) constant <3 x i32>, align 16
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(7) constant <3 x i32>, align 16
@_arg_4 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_4_t._arg_4_t, align 8
@_arg_6 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_6_t._arg_6_t, align 8
@_arg_7 = external dso_local local_unnamed_addr addrspace(10) global %struct._ZTS8_arg_7_t._arg_7_t, align 8

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11dim2_subscr"() local_unnamed_addr #0 comdat !kernel_arg_addr_space !5 !kernel_arg_access_qual !5 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !5 !kernel_arg_buffer_location !5 {
entry:
  %0 = load i32 addrspace(10)*, i32 addrspace(10)* addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_0_t._arg_0_t, %struct._ZTS8_arg_0_t._arg_0_t addrspace(10)* @_arg_0, i64 0, i32 0), align 8, !tbaa !6
  %agg.tmp1.sroa.0.sroa.2.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_2_t._arg_2_t, %struct._ZTS8_arg_2_t._arg_2_t addrspace(10)* @_arg_2, i64 0, i32 0, i32 0, i32 0, i64 1), align 8
  %agg.tmp2.sroa.0.sroa.0.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_3_t._arg_3_t, %struct._ZTS8_arg_3_t._arg_3_t addrspace(10)* @_arg_3, i64 0, i32 0, i32 0, i32 0, i64 0), align 8
  %agg.tmp2.sroa.0.sroa.2.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_3_t._arg_3_t, %struct._ZTS8_arg_3_t._arg_3_t addrspace(10)* @_arg_3, i64 0, i32 0, i32 0, i32 0, i64 1), align 8
  %1 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInGlobalInvocationId, align 16, !noalias !11
  %2 = extractelement <3 x i32> %1, i64 1
  %conv.i.i.i.i.i.i = zext i32 %2 to i64
  %3 = extractelement <3 x i32> %1, i64 0
  %conv.i.i1.i.i.i.i = zext i32 %3 to i64
  %4 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInNumWorkgroups, align 16, !noalias !20
  %5 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInWorkgroupSize, align 16, !noalias !20
  %6 = extractelement <3 x i32> %4, i64 0
  %7 = extractelement <3 x i32> %5, i64 0
  %mul.i.i3.i.i.i.i = mul i32 %6, %2
  %mul.i.i.i = mul i32 %mul.i.i3.i.i.i.i, %7
  %add6.i.i.i.i = add i64 %agg.tmp2.sroa.0.sroa.0.0.copyload, %conv.i.i.i.i.i.i
  %mul.1.i.i.i.i = mul i64 %add6.i.i.i.i, %agg.tmp1.sroa.0.sroa.2.0.copyload
  %add.1.i.i.i.i = add i64 %agg.tmp2.sroa.0.sroa.2.0.copyload, %conv.i.i1.i.i.i.i
  %add6.1.i.i.i.i = add i64 %add.1.i.i.i.i, %mul.1.i.i.i.i
  %ptridx.i.i.i = getelementptr inbounds i32, i32 addrspace(10)* %0, i64 %add6.1.i.i.i.i
  %call3.ascast.i.i = addrspacecast i32 addrspace(10)* %ptridx.i.i.i to i32*
  %8 = load i32, i32* %call3.ascast.i.i, align 4, !tbaa !25
  %sub8.i.i.i = add i32 %8, %3
  %conv5.i = add i32 %sub8.i.i.i, %mul.i.i.i
  store i32 %conv5.i, i32* %call3.ascast.i.i, align 4, !tbaa !25
  ret void
}

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11dim3_subscr"() local_unnamed_addr #0 comdat !kernel_arg_addr_space !5 !kernel_arg_access_qual !5 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !5 !kernel_arg_buffer_location !5 {
entry:
  %0 = load i32 addrspace(10)*, i32 addrspace(10)* addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_4_t._arg_4_t, %struct._ZTS8_arg_4_t._arg_4_t addrspace(10)* @_arg_4, i64 0, i32 0), align 8, !tbaa !27
  %agg.tmp1.sroa.0.sroa.2.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_6_t._arg_6_t, %struct._ZTS8_arg_6_t._arg_6_t addrspace(10)* @_arg_6, i64 0, i32 0, i32 0, i32 0, i64 1), align 8
  %agg.tmp1.sroa.0.sroa.3.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_6_t._arg_6_t, %struct._ZTS8_arg_6_t._arg_6_t addrspace(10)* @_arg_6, i64 0, i32 0, i32 0, i32 0, i64 2), align 8
  %agg.tmp2.sroa.0.sroa.0.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_7_t._arg_7_t, %struct._ZTS8_arg_7_t._arg_7_t addrspace(10)* @_arg_7, i64 0, i32 0, i32 0, i32 0, i64 0), align 8
  %agg.tmp2.sroa.0.sroa.2.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_7_t._arg_7_t, %struct._ZTS8_arg_7_t._arg_7_t addrspace(10)* @_arg_7, i64 0, i32 0, i32 0, i32 0, i64 1), align 8
  %agg.tmp2.sroa.0.sroa.3.0.copyload = load i64, i64 addrspace(10)* getelementptr inbounds (%struct._ZTS8_arg_7_t._arg_7_t, %struct._ZTS8_arg_7_t._arg_7_t addrspace(10)* @_arg_7, i64 0, i32 0, i32 0, i32 0, i64 2), align 8
  %1 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInGlobalInvocationId, align 16, !noalias !29
  %2 = extractelement <3 x i32> %1, i64 2
  %conv.i.i.i.i.i.i = zext i32 %2 to i64
  %3 = extractelement <3 x i32> %1, i64 1
  %conv.i.i2.i.i.i.i = zext i32 %3 to i64
  %4 = extractelement <3 x i32> %1, i64 0
  %conv.i.i1.i.i.i.i = zext i32 %4 to i64
  %5 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInNumWorkgroups, align 16, !noalias !38
  %6 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInWorkgroupSize, align 16, !noalias !38
  %7 = extractelement <3 x i32> %5, i64 1
  %8 = extractelement <3 x i32> %6, i64 1
  %9 = mul <3 x i32> %6, %5
  %mul.i.i3.i.i.i.i = extractelement <3 x i32> %9, i64 0
  %mul.i.i6.i.i.i.i = mul i32 %7, %2
  %mul.i.i.i = mul i32 %mul.i.i6.i.i.i.i, %8
  %sub11.i.i.i = add i32 %mul.i.i.i, %3
  %add.i.i.i = mul i32 %mul.i.i3.i.i.i.i, %sub11.i.i.i
  %add6.i.i.i.i = add i64 %agg.tmp2.sroa.0.sroa.0.0.copyload, %conv.i.i.i.i.i.i
  %mul.1.i.i.i.i = mul i64 %add6.i.i.i.i, %agg.tmp1.sroa.0.sroa.2.0.copyload
  %add.1.i.i.i.i = add i64 %agg.tmp2.sroa.0.sroa.2.0.copyload, %conv.i.i2.i.i.i.i
  %add6.1.i.i.i.i = add i64 %add.1.i.i.i.i, %mul.1.i.i.i.i
  %mul.2.i.i.i.i = mul i64 %add6.1.i.i.i.i, %agg.tmp1.sroa.0.sroa.3.0.copyload
  %add.2.i.i.i.i = add i64 %agg.tmp2.sroa.0.sroa.3.0.copyload, %conv.i.i1.i.i.i.i
  %add6.2.i.i.i.i = add i64 %add.2.i.i.i.i, %mul.2.i.i.i.i
  %ptridx.i.i.i = getelementptr inbounds i32, i32 addrspace(10)* %0, i64 %add6.2.i.i.i.i
  %call3.ascast.i.i = addrspacecast i32 addrspace(10)* %ptridx.i.i.i to i32*
  %10 = load i32, i32* %call3.ascast.i.i, align 4, !tbaa !25
  %add20.i.i.i = add i32 %10, %4
  %conv7.i = add i32 %add20.i.i.i, %add.i.i.i
  store i32 %conv7.i, i32* %call3.ascast.i.i, align 4, !tbaa !25
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
!4 = !{!"clang version 12.0.0 (git@github.com:tadeaustria/llvm.git ac95c393759173a72c2f81be2d0b74bc2a285d7c)"}
!5 = !{}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS8_arg_0_t", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !14, !16, !18}
!12 = distinct !{!12, !13, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv: %agg.result"}
!13 = distinct !{!13, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv"}
!14 = distinct !{!14, !15, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v: %agg.result"}
!15 = distinct !{!15, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v"}
!16 = distinct !{!16, !17, !"_ZN2cl4sycl6detail7Builder7getItemILi2ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!17 = distinct !{!17, !"_ZN2cl4sycl6detail7Builder7getItemILi2ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!18 = distinct !{!18, !19, !"_ZN2cl4sycl6detail7Builder10getElementILi2ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!19 = distinct !{!19, !"_ZN2cl4sycl6detail7Builder10getElementILi2ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!20 = !{!21, !23, !16, !18}
!21 = distinct !{!21, !22, !"_ZN7__spirv21InitSizesSTGlobalSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv: %agg.result"}
!22 = distinct !{!22, !"_ZN7__spirv21InitSizesSTGlobalSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv"}
!23 = distinct !{!23, !24, !"_ZN7__spirvL14initGlobalSizeILi2EN2cl4sycl5rangeILi2EEEEET0_v: %agg.result"}
!24 = distinct !{!24, !"_ZN7__spirvL14initGlobalSizeILi2EN2cl4sycl5rangeILi2EEEEET0_v"}
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !9, i64 0}
!27 = !{!28, !8, i64 0}
!28 = !{!"_ZTS8_arg_4_t", !8, i64 0}
!29 = !{!30, !32, !34, !36}
!30 = distinct !{!30, !31, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi3EN2cl4sycl2idILi3EEEE8initSizeEv: %agg.result"}
!31 = distinct !{!31, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi3EN2cl4sycl2idILi3EEEE8initSizeEv"}
!32 = distinct !{!32, !33, !"_ZN7__spirvL22initGlobalInvocationIdILi3EN2cl4sycl2idILi3EEEEET0_v: %agg.result"}
!33 = distinct !{!33, !"_ZN7__spirvL22initGlobalInvocationIdILi3EN2cl4sycl2idILi3EEEEET0_v"}
!34 = distinct !{!34, !35, !"_ZN2cl4sycl6detail7Builder7getItemILi3ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!35 = distinct !{!35, !"_ZN2cl4sycl6detail7Builder7getItemILi3ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!36 = distinct !{!36, !37, !"_ZN2cl4sycl6detail7Builder10getElementILi3ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!37 = distinct !{!37, !"_ZN2cl4sycl6detail7Builder10getElementILi3ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!38 = !{!39, !41, !34, !36}
!39 = distinct !{!39, !40, !"_ZN7__spirv21InitSizesSTGlobalSizeILi3EN2cl4sycl5rangeILi3EEEE8initSizeEv: %agg.result"}
!40 = distinct !{!40, !"_ZN7__spirv21InitSizesSTGlobalSizeILi3EN2cl4sycl5rangeILi3EEEE8initSizeEv"}
!41 = distinct !{!41, !42, !"_ZN7__spirvL14initGlobalSizeILi3EN2cl4sycl5rangeILi3EEEEET0_v: %agg.result"}
!42 = distinct !{!42, !"_ZN7__spirvL14initGlobalSizeILi3EN2cl4sycl5rangeILi3EEEEET0_v"}