Ł
ä“
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
}
ResourceApplyGradientDescent
var

alpha"T

delta"T" 
Ttype:
2	"
use_lockingbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"train*1.15.02v1.15.0-rc3-22-g590d6eef7e8Ńī
p
dense_inputPlaceholder*
shape:’’’’’’’’’*
dtype0*(
_output_shapes
:’’’’’’’’’

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ż[¾*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *ż[>*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ķ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	
Ī
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
į
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	

dense/kernelVarHandleOp*
shape:	*
shared_namedense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:


dense/biasVarHandleOp*
shape:*
shared_name
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	
r
dense/MatMulMatMuldense_inputdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
S

dense/TanhTanhdense/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄæ*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ?*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ņ
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
č
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
Ś
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

dense_1/kernelVarHandleOp*
shape
:*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

dense_1/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

dense_1/biasVarHandleOp*
shape:*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:
u
dense_1/MatMulMatMul
dense/Tanhdense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’

dense_1_targetPlaceholder*%
shape:’’’’’’’’’’’’’’’’’’*
dtype0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
x
totalVarHandleOp*
shape: *
shared_nametotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
x
countVarHandleOp*
shape: *
shared_namecount*
_class

loc:@count*
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
x
-metrics/categorical_accuracy/ArgMax/dimensionConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

#metrics/categorical_accuracy/ArgMaxArgMaxdense_1_target-metrics/categorical_accuracy/ArgMax/dimension*
T0*#
_output_shapes
:’’’’’’’’’
z
/metrics/categorical_accuracy/ArgMax_1/dimensionConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

%metrics/categorical_accuracy/ArgMax_1ArgMaxdense_1/BiasAdd/metrics/categorical_accuracy/ArgMax_1/dimension*
T0*#
_output_shapes
:’’’’’’’’’
„
"metrics/categorical_accuracy/EqualEqual#metrics/categorical_accuracy/ArgMax%metrics/categorical_accuracy/ArgMax_1*
T0	*#
_output_shapes
:’’’’’’’’’

!metrics/categorical_accuracy/CastCast"metrics/categorical_accuracy/Equal*

SrcT0
*#
_output_shapes
:’’’’’’’’’*

DstT0
l
"metrics/categorical_accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:

 metrics/categorical_accuracy/SumSum!metrics/categorical_accuracy/Cast"metrics/categorical_accuracy/Const*
T0*
_output_shapes
: 
}
0metrics/categorical_accuracy/AssignAddVariableOpAssignAddVariableOptotal metrics/categorical_accuracy/Sum*
dtype0
æ
+metrics/categorical_accuracy/ReadVariableOpReadVariableOptotal1^metrics/categorical_accuracy/AssignAddVariableOp!^metrics/categorical_accuracy/Sum*
dtype0*
_output_shapes
: 
m
!metrics/categorical_accuracy/SizeSize!metrics/categorical_accuracy/Cast*
T0*
_output_shapes
: 
~
#metrics/categorical_accuracy/Cast_1Cast!metrics/categorical_accuracy/Size*

SrcT0*
_output_shapes
: *

DstT0
µ
2metrics/categorical_accuracy/AssignAddVariableOp_1AssignAddVariableOpcount#metrics/categorical_accuracy/Cast_11^metrics/categorical_accuracy/AssignAddVariableOp*
dtype0
Ó
-metrics/categorical_accuracy/ReadVariableOp_1ReadVariableOpcount1^metrics/categorical_accuracy/AssignAddVariableOp3^metrics/categorical_accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
©
6metrics/categorical_accuracy/div_no_nan/ReadVariableOpReadVariableOptotal3^metrics/categorical_accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
«
8metrics/categorical_accuracy/div_no_nan/ReadVariableOp_1ReadVariableOpcount3^metrics/categorical_accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
Ę
'metrics/categorical_accuracy/div_no_nanDivNoNan6metrics/categorical_accuracy/div_no_nan/ReadVariableOp8metrics/categorical_accuracy/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
{
%metrics/categorical_accuracy/IdentityIdentity'metrics/categorical_accuracy/div_no_nan*
T0*
_output_shapes
: 
\
loss/dense_1_loss/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
8loss/dense_1_loss/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
x
9loss/dense_1_loss/softmax_cross_entropy_with_logits/ShapeShapedense_1/BiasAdd*
T0*
_output_shapes
:
|
:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
z
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_1Shapedense_1/BiasAdd*
T0*
_output_shapes
:
{
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ö
7loss/dense_1_loss/softmax_cross_entropy_with_logits/SubSub:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_19loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
®
?loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/beginPack7loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub*
T0*
N*
_output_shapes
:

>loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
²
9loss/dense_1_loss/softmax_cross_entropy_with_logits/SliceSlice;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_1?loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/begin>loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:

Closs/dense_1_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

?loss/dense_1_loss/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
µ
:loss/dense_1_loss/softmax_cross_entropy_with_logits/concatConcatV2Closs/dense_1_loss/softmax_cross_entropy_with_logits/concat/values_09loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice?loss/dense_1_loss/softmax_cross_entropy_with_logits/concat/axis*
T0*
N*
_output_shapes
:
Ī
;loss/dense_1_loss/softmax_cross_entropy_with_logits/ReshapeReshapedense_1/BiasAdd:loss/dense_1_loss/softmax_cross_entropy_with_logits/concat*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
|
:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
y
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_2Shapedense_1_target*
T0*
_output_shapes
:
}
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ś
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1Sub:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_2;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
²
Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1*
T0*
N*
_output_shapes
:

@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ø
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1Slice;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_2Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/begin@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:

Eloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

Aloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
½
<loss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Eloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/values_0;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1Aloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/axis*
T0*
N*
_output_shapes
:
Ń
=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_1_target<loss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

3loss/dense_1_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits;loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:’’’’’’’’’:’’’’’’’’’’’’’’’’’’
}
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ų
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2Sub8loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 

Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
±
@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2*
T0*
N*
_output_shapes
:
¶
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2Slice9loss/dense_1_loss/softmax_cross_entropy_with_logits/ShapeAloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/begin@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
č
=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape3loss/dense_1_loss/softmax_cross_entropy_with_logits;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2*
T0*#
_output_shapes
:’’’’’’’’’
k
&loss/dense_1_loss/weighted_loss/Cast/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Tloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
Ą
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:

Rloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
j
bloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp

Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2c^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:
ė
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
÷
;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:’’’’’’’’’
Ė
1loss/dense_1_loss/weighted_loss/broadcast_weightsMul&loss/dense_1_loss/weighted_loss/Cast/x;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:’’’’’’’’’
Ź
#loss/dense_1_loss/weighted_loss/MulMul=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_21loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:’’’’’’’’’
c
loss/dense_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
}
loss/dense_1_loss/SumSum#loss/dense_1_loss/weighted_loss/Mulloss/dense_1_loss/Const_1*
T0*
_output_shapes
: 
l
loss/dense_1_loss/num_elementsSize#loss/dense_1_loss/weighted_loss/Mul*
T0*
_output_shapes
: 
{
#loss/dense_1_loss/num_elements/CastCastloss/dense_1_loss/num_elements*

SrcT0*
_output_shapes
: *

DstT0
\
loss/dense_1_loss/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
q
loss/dense_1_loss/Sum_1Sumloss/dense_1_loss/Sumloss/dense_1_loss/Const_2*
T0*
_output_shapes
: 

loss/dense_1_loss/valueDivNoNanloss/dense_1_loss/Sum_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/dense_1_loss/value*
T0*
_output_shapes
: 
q
iter/Initializer/zerosConst*
value	B	 R *
_class
	loc:@iter*
dtype0	*
_output_shapes
: 
u
iterVarHandleOp*
shape: *
shared_nameiter*
_class
	loc:@iter*
dtype0	*
_output_shapes
: 
Y
%iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpiter*
_output_shapes
: 
J
iter/AssignAssignVariableOpiteriter/Initializer/zeros*
dtype0	
U
iter/Read/ReadVariableOpReadVariableOpiter*
dtype0	*
_output_shapes
: 
i
&training/SGD/gradients/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
o
*training/SGD/gradients/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¢
%training/SGD/gradients/gradients/FillFill&training/SGD/gradients/gradients/Shape*training/SGD/gradients/gradients/grad_ys_0*
T0*
_output_shapes
: 

2training/SGD/gradients/gradients/loss/mul_grad/MulMul%training/SGD/gradients/gradients/Fillloss/dense_1_loss/value*
T0*
_output_shapes
: 

4training/SGD/gradients/gradients/loss/mul_grad/Mul_1Mul%training/SGD/gradients/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 

Ctraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

Etraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
¬
Straining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/ShapeEtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Š
Htraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nanDivNoNan4training/SGD/gradients/gradients/loss/mul_grad/Mul_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 

Atraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/SumSumHtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nanStraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: 
ł
Etraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/ReshapeReshapeAtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/SumCtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Shape*
T0*
_output_shapes
: 

Atraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/NegNegloss/dense_1_loss/Sum_1*
T0*
_output_shapes
: 
ß
Jtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_1DivNoNanAtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Neg#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
č
Jtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_2DivNoNanJtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
ė
Atraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/mulMul4training/SGD/gradients/gradients/loss/mul_grad/Mul_1Jtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 

Ctraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Sum_1SumAtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/mulUtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: 
’
Gtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Reshape_1ReshapeCtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Sum_1Etraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/Shape_1*
T0*
_output_shapes
: 

Ktraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 

Etraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/ReshapeReshapeEtraining/SGD/gradients/gradients/loss/dense_1_loss/value_grad/ReshapeKtraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
: 

Ctraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
÷
Btraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/TileTileEtraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/ReshapeCtraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/Const*
T0*
_output_shapes
: 

Itraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

Ctraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/ReshapeReshapeBtraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_1_grad/TileItraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/Reshape/shape*
T0*
_output_shapes
:

Atraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/ShapeShape#loss/dense_1_loss/weighted_loss/Mul*
T0*
_output_shapes
:
ž
@training/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/TileTileCtraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/ReshapeAtraining/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/Shape*
T0*#
_output_shapes
:’’’’’’’’’
¼
Otraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ShapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:
²
Qtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1Shape1loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*
_output_shapes
:
Š
_training/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsOtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ShapeQtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
÷
Mtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/MulMul@training/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/Tile1loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:’’’’’’’’’
§
Mtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/SumSumMtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul_training/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
Ŗ
Qtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ReshapeReshapeMtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/SumOtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape*
T0*#
_output_shapes
:’’’’’’’’’

Otraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul_1Mul=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2@training/SGD/gradients/gradients/loss/dense_1_loss/Sum_grad/Tile*
T0*#
_output_shapes
:’’’’’’’’’
­
Otraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Sum_1SumOtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul_1atraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
°
Straining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshape_1ReshapeOtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Sum_1Qtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1*
T0*#
_output_shapes
:’’’’’’’’’
Ģ
itraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape3loss/dense_1_loss/softmax_cross_entropy_with_logits*
T0*
_output_shapes
:
ā
ktraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapeQtraining/SGD/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshapeitraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*#
_output_shapes
:’’’’’’’’’
Ŗ
+training/SGD/gradients/gradients/zeros_like	ZerosLike5loss/dense_1_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
³
htraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
ū
dtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsktraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapehtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
¼
]training/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mulMuldtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims5loss/dense_1_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
ź
dtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax;loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

]training/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/NegNegdtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
µ
jtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
’
ftraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsktraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapejtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:’’’’’’’’’
č
_training/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mul_1Mulftraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1]training/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
¦
gtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_1/BiasAdd*
T0*
_output_shapes
:
ī
itraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshape]training/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mulgtraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*'
_output_shapes
:’’’’’’’’’
ą
Atraining/SGD/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGraditraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
_output_shapes
:

;training/SGD/gradients/gradients/dense_1/MatMul_grad/MatMulMatMulitraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_1/MatMul/ReadVariableOp*
transpose_b(*
T0*'
_output_shapes
:’’’’’’’’’
ś
=training/SGD/gradients/gradients/dense_1/MatMul_grad/MatMul_1MatMul
dense/Tanhitraining/SGD/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
_output_shapes

:*
transpose_a(
Ą
9training/SGD/gradients/gradients/dense/Tanh_grad/TanhGradTanhGrad
dense/Tanh;training/SGD/gradients/gradients/dense_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:’’’’’’’’’
®
?training/SGD/gradients/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad9training/SGD/gradients/gradients/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes
:
į
9training/SGD/gradients/gradients/dense/MatMul_grad/MatMulMatMul9training/SGD/gradients/gradients/dense/Tanh_grad/TanhGraddense/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:’’’’’’’’’
Ź
;training/SGD/gradients/gradients/dense/MatMul_grad/MatMul_1MatMuldense_input9training/SGD/gradients/gradients/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes
:	*
transpose_a(

,training/SGD/decay/Initializer/initial_valueConst*
valueB
 *    *%
_class
loc:@training/SGD/decay*
dtype0*
_output_shapes
: 

training/SGD/decayVarHandleOp*
shape: *#
shared_nametraining/SGD/decay*%
_class
loc:@training/SGD/decay*
dtype0*
_output_shapes
: 
u
3training/SGD/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/SGD/decay*
_output_shapes
: 
|
training/SGD/decay/AssignAssignVariableOptraining/SGD/decay,training/SGD/decay/Initializer/initial_value*
dtype0
q
&training/SGD/decay/Read/ReadVariableOpReadVariableOptraining/SGD/decay*
dtype0*
_output_shapes
: 
Ø
4training/SGD/learning_rate/Initializer/initial_valueConst*
valueB
 *
×#<*-
_class#
!loc:@training/SGD/learning_rate*
dtype0*
_output_shapes
: 
·
training/SGD/learning_rateVarHandleOp*
shape: *+
shared_nametraining/SGD/learning_rate*-
_class#
!loc:@training/SGD/learning_rate*
dtype0*
_output_shapes
: 

;training/SGD/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/SGD/learning_rate*
_output_shapes
: 

!training/SGD/learning_rate/AssignAssignVariableOptraining/SGD/learning_rate4training/SGD/learning_rate/Initializer/initial_value*
dtype0

.training/SGD/learning_rate/Read/ReadVariableOpReadVariableOptraining/SGD/learning_rate*
dtype0*
_output_shapes
: 

/training/SGD/momentum/Initializer/initial_valueConst*
valueB
 *    *(
_class
loc:@training/SGD/momentum*
dtype0*
_output_shapes
: 
Ø
training/SGD/momentumVarHandleOp*
shape: *&
shared_nametraining/SGD/momentum*(
_class
loc:@training/SGD/momentum*
dtype0*
_output_shapes
: 
{
6training/SGD/momentum/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/SGD/momentum*
_output_shapes
: 

training/SGD/momentum/AssignAssignVariableOptraining/SGD/momentum/training/SGD/momentum/Initializer/initial_value*
dtype0
w
)training/SGD/momentum/Read/ReadVariableOpReadVariableOptraining/SGD/momentum*
dtype0*
_output_shapes
: 
w
$training/SGD/Identity/ReadVariableOpReadVariableOptraining/SGD/learning_rate*
dtype0*
_output_shapes
: 
h
training/SGD/IdentityIdentity$training/SGD/Identity/ReadVariableOp*
T0*
_output_shapes
: 
t
&training/SGD/Identity_1/ReadVariableOpReadVariableOptraining/SGD/momentum*
dtype0*
_output_shapes
: 
l
training/SGD/Identity_1Identity&training/SGD/Identity_1/ReadVariableOp*
T0*
_output_shapes
: 

Atraining/SGD/SGD/update_dense/kernel/ResourceApplyGradientDescentResourceApplyGradientDescentdense/kerneltraining/SGD/Identity;training/SGD/gradients/gradients/dense/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@dense/kernel
ž
?training/SGD/SGD/update_dense/bias/ResourceApplyGradientDescentResourceApplyGradientDescent
dense/biastraining/SGD/Identity?training/SGD/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense/bias

Ctraining/SGD/SGD/update_dense_1/kernel/ResourceApplyGradientDescentResourceApplyGradientDescentdense_1/kerneltraining/SGD/Identity=training/SGD/gradients/gradients/dense_1/MatMul_grad/MatMul_1*
use_locking(*
T0*!
_class
loc:@dense_1/kernel

Atraining/SGD/SGD/update_dense_1/bias/ResourceApplyGradientDescentResourceApplyGradientDescentdense_1/biastraining/SGD/IdentityAtraining/SGD/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense_1/bias
č
training/SGD/SGD/ConstConst@^training/SGD/SGD/update_dense/bias/ResourceApplyGradientDescentB^training/SGD/SGD/update_dense/kernel/ResourceApplyGradientDescentB^training/SGD/SGD/update_dense_1/bias/ResourceApplyGradientDescentD^training/SGD/SGD/update_dense_1/kernel/ResourceApplyGradientDescent*
value	B	 R*
dtype0	*
_output_shapes
: 
f
$training/SGD/SGD/AssignAddVariableOpAssignAddVariableOpitertraining/SGD/SGD/Const*
dtype0	

training/SGD/SGD/ReadVariableOpReadVariableOpiter%^training/SGD/SGD/AssignAddVariableOp@^training/SGD/SGD/update_dense/bias/ResourceApplyGradientDescentB^training/SGD/SGD/update_dense/kernel/ResourceApplyGradientDescentB^training/SGD/SGD/update_dense_1/bias/ResourceApplyGradientDescentD^training/SGD/SGD/update_dense_1/kernel/ResourceApplyGradientDescent*
dtype0	*
_output_shapes
: 
O
training_1/group_depsNoOp	^loss/mul%^training/SGD/SGD/AssignAddVariableOp
Z
ConstConst"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 

RestoreV2/tensor_namesConst"/device:CPU:0*«
value”BB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

RestoreV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:
®
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes

2	*4
_output_shapes"
 ::::::::
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
G
AssignVariableOpAssignVariableOp
dense/biasIdentity*
dtype0
F

Identity_1IdentityRestoreV2:1*
T0*
_output_shapes
:
M
AssignVariableOp_1AssignVariableOpdense/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:2*
T0*
_output_shapes
:
M
AssignVariableOp_2AssignVariableOpdense_1/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:3*
T0*
_output_shapes
:
O
AssignVariableOp_3AssignVariableOpdense_1/kernel
Identity_3*
dtype0
F

Identity_4IdentityRestoreV2:4*
T0*
_output_shapes
:
S
AssignVariableOp_4AssignVariableOptraining/SGD/decay
Identity_4*
dtype0
F

Identity_5IdentityRestoreV2:5*
T0	*
_output_shapes
:
E
AssignVariableOp_5AssignVariableOpiter
Identity_5*
dtype0	
F

Identity_6IdentityRestoreV2:6*
T0*
_output_shapes
:
[
AssignVariableOp_6AssignVariableOptraining/SGD/learning_rate
Identity_6*
dtype0
F

Identity_7IdentityRestoreV2:7*
T0*
_output_shapes
:
V
AssignVariableOp_7AssignVariableOptraining/SGD/momentum
Identity_7*
dtype0
G
VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
N
VarIsInitializedOp_1VarIsInitializedOp
dense/bias*
_output_shapes
: 
I
VarIsInitializedOp_2VarIsInitializedOpcount*
_output_shapes
: 
H
VarIsInitializedOp_3VarIsInitializedOpiter*
_output_shapes
: 
V
VarIsInitializedOp_4VarIsInitializedOptraining/SGD/decay*
_output_shapes
: 
R
VarIsInitializedOp_5VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
^
VarIsInitializedOp_6VarIsInitializedOptraining/SGD/learning_rate*
_output_shapes
: 
Y
VarIsInitializedOp_7VarIsInitializedOptraining/SGD/momentum*
_output_shapes
: 
P
VarIsInitializedOp_8VarIsInitializedOpdense_1/bias*
_output_shapes
: 
P
VarIsInitializedOp_9VarIsInitializedOpdense/kernel*
_output_shapes
: 
ļ
initNoOp^count/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^iter/Assign^total/Assign^training/SGD/decay/Assign"^training/SGD/learning_rate/Assign^training/SGD/momentum/Assign
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 

StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_45fb2b0b0ac34a75b2c3145fd3a858f3/part*
dtype0*
_output_shapes
: 
f

StringJoin
StringJoinConst_2StringJoin/inputs_1"/device:CPU:0*
N*
_output_shapes
: 
L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 

SaveV2/tensor_namesConst"/device:CPU:0*«
value”BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
}
SaveV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:

SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpiter/Read/ReadVariableOp&training/SGD/decay/Read/ReadVariableOp.training/SGD/learning_rate/Read/ReadVariableOp)training/SGD/momentum/Read/ReadVariableOp"/device:CPU:0*
dtypes

2	
h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
|
ShardedFilename_1ShardedFilename
StringJoinShardedFilename_1/shard
num_shards"/device:CPU:0*
_output_shapes
: 

SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:
q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

SaveV2_1SaveV2ShardedFilename_1SaveV2_1/tensor_namesSaveV2_1/shape_and_slicesConst_1"/device:CPU:0*
dtypes
2
£
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilenameShardedFilename_1^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
h
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixesConst_2"/device:CPU:0
d

Identity_8IdentityConst_2^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
W
div_no_nan/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
Y
div_no_nan/ReadVariableOp_1ReadVariableOpcount*
dtype0*
_output_shapes
: 
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
C

Identity_9Identity
div_no_nan*
T0*
_output_shapes
: 

metric_op_wrapperConst3^metrics/categorical_accuracy/AssignAddVariableOp_1*
valueB *
dtype0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
ų
save/SaveV2/tensor_namesConst*«
value”BB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
s
save/SaveV2/shape_and_slicesConst*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp&training/SGD/decay/Read/ReadVariableOpiter/Read/ReadVariableOp.training/SGD/learning_rate/Read/ReadVariableOp)training/SGD/momentum/Read/ReadVariableOp*
dtypes

2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*«
value”BB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:
Ā
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes

2	*4
_output_shapes"
 ::::::::
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
Q
save/AssignVariableOpAssignVariableOp
dense/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
W
save/AssignVariableOp_1AssignVariableOpdense/kernelsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
W
save/AssignVariableOp_2AssignVariableOpdense_1/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_3AssignVariableOpdense_1/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0*
_output_shapes
:
]
save/AssignVariableOp_4AssignVariableOptraining/SGD/decaysave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
T0	*
_output_shapes
:
O
save/AssignVariableOp_5AssignVariableOpitersave/Identity_5*
dtype0	
P
save/Identity_6Identitysave/RestoreV2:6*
T0*
_output_shapes
:
e
save/AssignVariableOp_6AssignVariableOptraining/SGD/learning_ratesave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:7*
T0*
_output_shapes
:
`
save/AssignVariableOp_7AssignVariableOptraining/SGD/momentumsave/Identity_7*
dtype0
ę
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7
,
init_1NoOp^count/Assign^total/Assign"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"ņ
trainable_variablesŚ×
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"Å
local_variables±®
U
total:0total/Assigntotal/Read/ReadVariableOp:0(2total/Initializer/zeros:0@H
U
count:0count/Assigncount/Read/ReadVariableOp:0(2count/Initializer/zeros:0@H"b
global_stepSQ
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H"
	variables
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H

training/SGD/decay:0training/SGD/decay/Assign(training/SGD/decay/Read/ReadVariableOp:0(2.training/SGD/decay/Initializer/initial_value:0H
Æ
training/SGD/learning_rate:0!training/SGD/learning_rate/Assign0training/SGD/learning_rate/Read/ReadVariableOp:0(26training/SGD/learning_rate/Initializer/initial_value:0H

training/SGD/momentum:0training/SGD/momentum/Assign+training/SGD/momentum/Read/ReadVariableOp:0(21training/SGD/momentum/Initializer/initial_value:0H*Q
__saved_model_train_op75
__saved_model_train_op
training_1/group_deps*
trainö
B
dense_1_target0
dense_1_target:0’’’’’’’’’’’’’’’’’’
4
dense_input%
dense_input:0’’’’’’’’’E
&metrics/categorical_accuracy/update_op
metric_op_wrapper:0 8
"metrics/categorical_accuracy/value
Identity_9:0 ?
predictions/dense_1(
dense_1/BiasAdd:0’’’’’’’’’
loss

loss/mul:0 tensorflow/supervised/training*@
__saved_model_init_op'%
__saved_model_init_op
init_1Āē
üĶ
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"eval*1.15.02v1.15.0-rc3-22-g590d6eef7e8ÜĀ
p
dense_inputPlaceholder*
shape:’’’’’’’’’*
dtype0*(
_output_shapes
:’’’’’’’’’

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ż[¾*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *ż[>*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ķ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	
Ī
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
į
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	

dense/kernelVarHandleOp*
shape:	*
shared_namedense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:


dense/biasVarHandleOp*
shape:*
shared_name
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	
r
dense/MatMulMatMuldense_inputdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
S

dense/TanhTanhdense/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄæ*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ?*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ņ
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
č
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
Ś
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

dense_1/kernelVarHandleOp*
shape
:*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

dense_1/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

dense_1/biasVarHandleOp*
shape:*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:
u
dense_1/MatMulMatMul
dense/Tanhdense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’

dense_1_targetPlaceholder*%
shape:’’’’’’’’’’’’’’’’’’*
dtype0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
x
totalVarHandleOp*
shape: *
shared_nametotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
x
countVarHandleOp*
shape: *
shared_namecount*
_class

loc:@count*
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
x
-metrics/categorical_accuracy/ArgMax/dimensionConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

#metrics/categorical_accuracy/ArgMaxArgMaxdense_1_target-metrics/categorical_accuracy/ArgMax/dimension*
T0*#
_output_shapes
:’’’’’’’’’
z
/metrics/categorical_accuracy/ArgMax_1/dimensionConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 

%metrics/categorical_accuracy/ArgMax_1ArgMaxdense_1/BiasAdd/metrics/categorical_accuracy/ArgMax_1/dimension*
T0*#
_output_shapes
:’’’’’’’’’
„
"metrics/categorical_accuracy/EqualEqual#metrics/categorical_accuracy/ArgMax%metrics/categorical_accuracy/ArgMax_1*
T0	*#
_output_shapes
:’’’’’’’’’

!metrics/categorical_accuracy/CastCast"metrics/categorical_accuracy/Equal*

SrcT0
*#
_output_shapes
:’’’’’’’’’*

DstT0
l
"metrics/categorical_accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:

 metrics/categorical_accuracy/SumSum!metrics/categorical_accuracy/Cast"metrics/categorical_accuracy/Const*
T0*
_output_shapes
: 
}
0metrics/categorical_accuracy/AssignAddVariableOpAssignAddVariableOptotal metrics/categorical_accuracy/Sum*
dtype0
æ
+metrics/categorical_accuracy/ReadVariableOpReadVariableOptotal1^metrics/categorical_accuracy/AssignAddVariableOp!^metrics/categorical_accuracy/Sum*
dtype0*
_output_shapes
: 
m
!metrics/categorical_accuracy/SizeSize!metrics/categorical_accuracy/Cast*
T0*
_output_shapes
: 
~
#metrics/categorical_accuracy/Cast_1Cast!metrics/categorical_accuracy/Size*

SrcT0*
_output_shapes
: *

DstT0
µ
2metrics/categorical_accuracy/AssignAddVariableOp_1AssignAddVariableOpcount#metrics/categorical_accuracy/Cast_11^metrics/categorical_accuracy/AssignAddVariableOp*
dtype0
Ó
-metrics/categorical_accuracy/ReadVariableOp_1ReadVariableOpcount1^metrics/categorical_accuracy/AssignAddVariableOp3^metrics/categorical_accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
©
6metrics/categorical_accuracy/div_no_nan/ReadVariableOpReadVariableOptotal3^metrics/categorical_accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
«
8metrics/categorical_accuracy/div_no_nan/ReadVariableOp_1ReadVariableOpcount3^metrics/categorical_accuracy/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
Ę
'metrics/categorical_accuracy/div_no_nanDivNoNan6metrics/categorical_accuracy/div_no_nan/ReadVariableOp8metrics/categorical_accuracy/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
{
%metrics/categorical_accuracy/IdentityIdentity'metrics/categorical_accuracy/div_no_nan*
T0*
_output_shapes
: 
\
loss/dense_1_loss/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
8loss/dense_1_loss/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
x
9loss/dense_1_loss/softmax_cross_entropy_with_logits/ShapeShapedense_1/BiasAdd*
T0*
_output_shapes
:
|
:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
z
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_1Shapedense_1/BiasAdd*
T0*
_output_shapes
:
{
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ö
7loss/dense_1_loss/softmax_cross_entropy_with_logits/SubSub:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_19loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
®
?loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/beginPack7loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub*
T0*
N*
_output_shapes
:

>loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
²
9loss/dense_1_loss/softmax_cross_entropy_with_logits/SliceSlice;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_1?loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/begin>loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:

Closs/dense_1_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

?loss/dense_1_loss/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
µ
:loss/dense_1_loss/softmax_cross_entropy_with_logits/concatConcatV2Closs/dense_1_loss/softmax_cross_entropy_with_logits/concat/values_09loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice?loss/dense_1_loss/softmax_cross_entropy_with_logits/concat/axis*
T0*
N*
_output_shapes
:
Ī
;loss/dense_1_loss/softmax_cross_entropy_with_logits/ReshapeReshapedense_1/BiasAdd:loss/dense_1_loss/softmax_cross_entropy_with_logits/concat*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
|
:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
y
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_2Shapedense_1_target*
T0*
_output_shapes
:
}
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ś
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1Sub:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_2;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
²
Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1*
T0*
N*
_output_shapes
:

@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ø
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1Slice;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_2Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/begin@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:

Eloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

Aloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
½
<loss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Eloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/values_0;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1Aloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/axis*
T0*
N*
_output_shapes
:
Ń
=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_1_target<loss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

3loss/dense_1_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits;loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:’’’’’’’’’:’’’’’’’’’’’’’’’’’’
}
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ų
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2Sub8loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 

Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
±
@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2*
T0*
N*
_output_shapes
:
¶
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2Slice9loss/dense_1_loss/softmax_cross_entropy_with_logits/ShapeAloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/begin@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
č
=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape3loss/dense_1_loss/softmax_cross_entropy_with_logits;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2*
T0*#
_output_shapes
:’’’’’’’’’
k
&loss/dense_1_loss/weighted_loss/Cast/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Tloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
Ą
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:

Rloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
j
bloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp

Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2c^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:
ė
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
÷
;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:’’’’’’’’’
Ė
1loss/dense_1_loss/weighted_loss/broadcast_weightsMul&loss/dense_1_loss/weighted_loss/Cast/x;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:’’’’’’’’’
Ź
#loss/dense_1_loss/weighted_loss/MulMul=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_21loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:’’’’’’’’’
c
loss/dense_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
}
loss/dense_1_loss/SumSum#loss/dense_1_loss/weighted_loss/Mulloss/dense_1_loss/Const_1*
T0*
_output_shapes
: 
l
loss/dense_1_loss/num_elementsSize#loss/dense_1_loss/weighted_loss/Mul*
T0*
_output_shapes
: 
{
#loss/dense_1_loss/num_elements/CastCastloss/dense_1_loss/num_elements*

SrcT0*
_output_shapes
: *

DstT0
\
loss/dense_1_loss/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
q
loss/dense_1_loss/Sum_1Sumloss/dense_1_loss/Sumloss/dense_1_loss/Const_2*
T0*
_output_shapes
: 

loss/dense_1_loss/valueDivNoNanloss/dense_1_loss/Sum_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/dense_1_loss/value*
T0*
_output_shapes
: 
q
iter/Initializer/zerosConst*
value	B	 R *
_class
	loc:@iter*
dtype0	*
_output_shapes
: 
u
iterVarHandleOp*
shape: *
shared_nameiter*
_class
	loc:@iter*
dtype0	*
_output_shapes
: 
Y
%iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpiter*
_output_shapes
: 
J
iter/AssignAssignVariableOpiteriter/Initializer/zeros*
dtype0	
U
iter/Read/ReadVariableOpReadVariableOpiter*
dtype0	*
_output_shapes
: 
(
evaluation/group_depsNoOp	^loss/mul
Z
ConstConst"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
ö
RestoreV2/tensor_namesConst"/device:CPU:0*
valueBB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
z
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:

	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes	
2	*(
_output_shapes
:::::
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
G
AssignVariableOpAssignVariableOp
dense/biasIdentity*
dtype0
F

Identity_1IdentityRestoreV2:1*
T0*
_output_shapes
:
M
AssignVariableOp_1AssignVariableOpdense/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:2*
T0*
_output_shapes
:
M
AssignVariableOp_2AssignVariableOpdense_1/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:3*
T0*
_output_shapes
:
O
AssignVariableOp_3AssignVariableOpdense_1/kernel
Identity_3*
dtype0
F

Identity_4IdentityRestoreV2:4*
T0	*
_output_shapes
:
E
AssignVariableOp_4AssignVariableOpiter
Identity_4*
dtype0	
N
VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
R
VarIsInitializedOp_1VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
N
VarIsInitializedOp_2VarIsInitializedOp
dense/bias*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpdense_1/bias*
_output_shapes
: 
I
VarIsInitializedOp_4VarIsInitializedOpcount*
_output_shapes
: 
H
VarIsInitializedOp_5VarIsInitializedOpiter*
_output_shapes
: 
I
VarIsInitializedOp_6VarIsInitializedOptotal*
_output_shapes
: 

initNoOp^count/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^iter/Assign^total/Assign
W
div_no_nan/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
Y
div_no_nan/ReadVariableOp_1ReadVariableOpcount*
dtype0*
_output_shapes
: 
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
C

Identity_5Identity
div_no_nan*
T0*
_output_shapes
: 

metric_op_wrapperConst3^metrics/categorical_accuracy/AssignAddVariableOp_1*
valueB *
dtype0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
é
save/SaveV2/tensor_namesConst*
valueBB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
m
save/SaveV2/shape_and_slicesConst*
valueBB B B B B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOpiter/Read/ReadVariableOp*
dtypes	
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
ū
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:
³
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes	
2	*(
_output_shapes
:::::
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
Q
save/AssignVariableOpAssignVariableOp
dense/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
W
save/AssignVariableOp_1AssignVariableOpdense/kernelsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
W
save/AssignVariableOp_2AssignVariableOpdense_1/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_3AssignVariableOpdense_1/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0	*
_output_shapes
:
O
save/AssignVariableOp_4AssignVariableOpitersave/Identity_4*
dtype0	

save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4
,
init_1NoOp^count/Assign^total/Assign"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"ņ
trainable_variablesŚ×
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"Å
local_variables±®
U
total:0total/Assigntotal/Read/ReadVariableOp:0(2total/Initializer/zeros:0@H
U
count:0count/Assigncount/Read/ReadVariableOp:0(2count/Initializer/zeros:0@H"b
global_stepSQ
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H"¹
	variables«Ø
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H*ū
evalņ
B
dense_1_target0
dense_1_target:0’’’’’’’’’’’’’’’’’’
4
dense_input%
dense_input:0’’’’’’’’’E
&metrics/categorical_accuracy/update_op
metric_op_wrapper:0 8
"metrics/categorical_accuracy/value
Identity_5:0 ?
predictions/dense_1(
dense_1/BiasAdd:0’’’’’’’’’
loss

loss/mul:0 tensorflow/supervised/eval*@
__saved_model_init_op'%
__saved_model_init_op
init_1ńe
­ż

:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*1.15.02v1.15.0-rc3-22-g590d6eef7e8øP
p
dense_inputPlaceholder*
shape:’’’’’’’’’*
dtype0*(
_output_shapes
:’’’’’’’’’

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ż[¾*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *ż[>*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ķ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	
Ī
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
į
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	

dense/kernelVarHandleOp*
shape:	*
shared_namedense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:


dense/biasVarHandleOp*
shape:*
shared_name
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	
r
dense/MatMulMatMuldense_inputdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
S

dense/TanhTanhdense/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄæ*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ?*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ņ
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
č
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
Ś
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

dense_1/kernelVarHandleOp*
shape
:*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

dense_1/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

dense_1/biasVarHandleOp*
shape:*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:
u
dense_1/MatMulMatMul
dense/Tanhdense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’
,
predict/group_depsNoOp^dense_1/BiasAdd
Z
ConstConst"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ė
RestoreV2/tensor_namesConst"/device:CPU:0*ń
valueēBäB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
x
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:

	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*$
_output_shapes
::::
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
G
AssignVariableOpAssignVariableOp
dense/biasIdentity*
dtype0
F

Identity_1IdentityRestoreV2:1*
T0*
_output_shapes
:
M
AssignVariableOp_1AssignVariableOpdense/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:2*
T0*
_output_shapes
:
M
AssignVariableOp_2AssignVariableOpdense_1/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:3*
T0*
_output_shapes
:
O
AssignVariableOp_3AssignVariableOpdense_1/kernel
Identity_3*
dtype0
N
VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
P
VarIsInitializedOp_1VarIsInitializedOpdense_1/bias*
_output_shapes
: 
R
VarIsInitializedOp_2VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
N
VarIsInitializedOp_3VarIsInitializedOp
dense/bias*
_output_shapes
: 
d
initNoOp^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
¾
save/SaveV2/tensor_namesConst*ń
valueēBäB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
k
save/SaveV2/shape_and_slicesConst*
valueBB B B B *
dtype0*
_output_shapes
:
õ
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Š
save/RestoreV2/tensor_namesConst"/device:CPU:0*ń
valueēBäB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
}
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
®
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*$
_output_shapes
::::
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
Q
save/AssignVariableOpAssignVariableOp
dense/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
W
save/AssignVariableOp_1AssignVariableOpdense/kernelsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
W
save/AssignVariableOp_2AssignVariableOpdense_1/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_3AssignVariableOpdense_1/kernelsave/Identity_3*
dtype0
~
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3

init_1NoOp"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"ņ
trainable_variablesŚ×
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"č
	variablesŚ×
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08*
serving_default
4
dense_input%
dense_input:0’’’’’’’’’3
dense_1(
dense_1/BiasAdd:0’’’’’’’’’tensorflow/serving/predict*@
__saved_model_init_op'%
__saved_model_init_op
init_1