Ïî%
½
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

	MirrorPad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	"&
modestring:
REFLECT	SYMMETRIC
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28ÇÌ!

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv4/kernel

'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
}
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:*
dtype0
s
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
l
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes	
:*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel

$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*(
_output_shapes
:*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel

$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*(
_output_shapes
:*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_13/kernel

$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*(
_output_shapes
:*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:*
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_14/kernel

$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:*
dtype0
u
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_14/bias
n
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes	
:*
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_15/kernel
~
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*'
_output_shapes
:@*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0

conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:@*
dtype0

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:*
dtype0
Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"ÅàÏÂÙéÂ)\÷Â
\
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"ÅàÏÂÙéÂ)\÷Â

NoOpNoOp
q
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*Ïp
valueÅpBÂp B»p
´
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer-9
layer_with_weights-1
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 

	keras_api

	keras_api

	keras_api

	keras_api

	keras_api

	keras_api
è
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
 layer_with_weights-5
 layer-8
!layer_with_weights-6
!layer-9
"layer_with_weights-7
"layer-10
#layer-11
$layer_with_weights-8
$layer-12
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
Ú
-layer_with_weights-0
-layer-0
.layer-1
/layer_with_weights-1
/layer-2
0layer_with_weights-2
0layer-3
1layer_with_weights-3
1layer-4
2layer_with_weights-4
2layer-5
3layer-6
4layer_with_weights-5
4layer-7
5layer_with_weights-6
5layer-8
6layer-9
7layer_with_weights-7
7layer-10
8layer_with_weights-8
8layer-11
9	variables
:trainable_variables
;regularization_losses
<	keras_api

=	keras_api

>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
X26
Y27
Z28
[29
\30
]31
^32
_33
`34
a35

>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
X26
Y27
Z28
[29
\30
]31
^32
_33
`34
a35
 
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
 
 
 
 
h

>kernel
?bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
h

@kernel
Abias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
R
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

Bkernel
Cbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
h

Dkernel
Ebias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api
k

Fkernel
Gbias
	variables
trainable_variables
regularization_losses
	keras_api
l

Hkernel
Ibias
	variables
trainable_variables
regularization_losses
	keras_api
l

Jkernel
Kbias
	variables
trainable_variables
regularization_losses
	keras_api
l

Lkernel
Mbias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

Nkernel
Obias
	variables
trainable_variables
regularization_losses
	keras_api

>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17

>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
)	variables
*trainable_variables
+regularization_losses
z

Pkernel
Pw
Qbias
Qb
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
V
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
z

Rkernel
Rw
Sbias
Sb
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
z

Tkernel
Tw
Ubias
Ub
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
z

Vkernel
Vw
Wbias
Wb
±	variables
²trainable_variables
³regularization_losses
´	keras_api
z

Xkernel
Xw
Ybias
Yb
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
V
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
z

Zkernel
Zw
[bias
[b
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
z

\kernel
\w
]bias
]b
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
V
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
z

^kernel
^w
_bias
_b
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
z

`kernel
`w
abias
ab
Í	variables
Îtrainable_variables
Ïregularization_losses
Ð	keras_api

P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
`16
a17

P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
`16
a17
 
²
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
9	variables
:trainable_variables
;regularization_losses
 
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_9/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_9/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_10/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_10/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_11/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_11/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_12/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_12/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_13/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_13/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_14/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_14/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_15/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_15/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_16/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_16/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_17/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_17/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 

>0
?1

>0
?1
 
²
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
g	variables
htrainable_variables
iregularization_losses

@0
A1

@0
A1
 
²
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
 
 
 
²
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
o	variables
ptrainable_variables
qregularization_losses

B0
C1

B0
C1
 
²
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
s	variables
ttrainable_variables
uregularization_losses

D0
E1

D0
E1
 
²
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
 
 
 
²
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
{	variables
|trainable_variables
}regularization_losses

F0
G1

F0
G1
 
´
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses

H0
I1

H0
I1
 
µ
ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses

J0
K1

J0
K1
 
µ
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses

L0
M1

L0
M1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses

N0
O1

N0
O1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
^
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
 
 
 
 
 
 
 
 

P0
Q1

P0
Q1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
 
 
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¥	variables
¦trainable_variables
§regularization_losses

R0
S1

R0
S1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
©	variables
ªtrainable_variables
«regularization_losses

T0
U1

T0
U1
 
µ
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
­	variables
®trainable_variables
¯regularization_losses

V0
W1

V0
W1
 
µ
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
±	variables
²trainable_variables
³regularization_losses

X0
Y1

X0
Y1
 
µ
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
µ	variables
¶trainable_variables
·regularization_losses
 
 
 
µ
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses

Z0
[1

Z0
[1
 
µ
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
½	variables
¾trainable_variables
¿regularization_losses

\0
]1

\0
]1
 
µ
ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
 
 
 
µ
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses

^0
_1

^0
_1
 
µ
Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses

`0
a1

`0
a1
 
µ
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
Í	variables
Îtrainable_variables
Ïregularization_losses
 
V
-0
.1
/2
03
14
25
36
47
58
69
710
811
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_content_imagePlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_style_imagePlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
Ô
StatefulPartitionedCallStatefulPartitionedCallserving_default_content_imageserving_default_style_imageConstConst_1block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/bias*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_10596
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
«
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOpConst_2*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_12690
Ô
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_12808Õñ
â
e
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8944

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ@
º
B__inference_model_2_layer_call_and_return_conditional_losses_10387
content_image
style_image!
tf_nn_bias_add_4_biasadd_bias!
tf_nn_bias_add_3_biasadd_bias'
encoder_10282:@
encoder_10284:@'
encoder_10286:@@
encoder_10288:@(
encoder_10290:@
encoder_10292:	)
encoder_10294:
encoder_10296:	)
encoder_10298:
encoder_10300:	)
encoder_10302:
encoder_10304:	)
encoder_10306:
encoder_10308:	)
encoder_10310:
encoder_10312:	)
encoder_10314:
encoder_10316:	)
decoder_10345:
decoder_10347:	)
decoder_10349:
decoder_10351:	)
decoder_10353:
decoder_10355:	)
decoder_10357:
decoder_10359:	)
decoder_10361:
decoder_10363:	)
decoder_10365:
decoder_10367:	(
decoder_10369:@
decoder_10371:@'
decoder_10373:@@
decoder_10375:@'
decoder_10377:@
decoder_10379:
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall¢!encoder/StatefulPartitionedCall_1n
tf.reverse_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_4/ReverseV2	ReverseV2style_image$tf.reverse_4/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
tf.reverse_3/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_3/ReverseV2	ReverseV2content_image$tf.reverse_3/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_4/strided_sliceStridedSlicetf.reverse_4/ReverseV2:output:07tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_3/strided_sliceStridedSlicetf.reverse_3/ReverseV2:output:07tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask±
tf.nn.bias_add_4/BiasAddBiasAdd1tf.__operators__.getitem_4/strided_slice:output:0tf_nn_bias_add_4_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
encoder/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0encoder_10282encoder_10284encoder_10286encoder_10288encoder_10290encoder_10292encoder_10294encoder_10296encoder_10298encoder_10300encoder_10302encoder_10304encoder_10306encoder_10308encoder_10310encoder_10312encoder_10314encoder_10316*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9504ü
!encoder/StatefulPartitionedCall_1StatefulPartitionedCall!tf.nn.bias_add_4/BiasAdd:output:0encoder_10282encoder_10284encoder_10286encoder_10288encoder_10290encoder_10292encoder_10294encoder_10296encoder_10298encoder_10300encoder_10302encoder_10304encoder_10306encoder_10308encoder_10310encoder_10312encoder_10314encoder_10316*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9504
ada_in_1/PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:3*encoder/StatefulPartitionedCall_1:output:3*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_ada_in_1_layer_call_and_return_conditional_losses_9594 
decoder/StatefulPartitionedCallStatefulPartitionedCall!ada_in_1/PartitionedCall:output:0decoder_10345decoder_10347decoder_10349decoder_10351decoder_10353decoder_10355decoder_10357decoder_10359decoder_10361decoder_10363decoder_10365decoder_10367decoder_10369decoder_10371decoder_10373decoder_10375decoder_10377decoder_10379*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_8984o
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÎ
(tf.clip_by_value_1/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Â
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity$tf.clip_by_value_1/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall"^encoder/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2F
!encoder/StatefulPartitionedCall_1!encoder/StatefulPartitionedCall_1:` \
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecontent_image:^Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namestyle_image: 

_output_shapes
:: 

_output_shapes
:
­g
ã
B__inference_encoder_layer_call_and_return_conditional_losses_11457

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Å
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Þ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¾
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentityblock1_conv1/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identityblock2_conv1/Relu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identityblock3_conv1/Relu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identityblock4_conv1/Relu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ºb
ã
B__inference_encoder_layer_call_and_return_conditional_losses_11603

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@®
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ì
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ï
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  x
IdentityIdentityblock1_conv1/Relu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{

Identity_1Identityblock2_conv1/Relu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿy

Identity_2Identityblock3_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@y

Identity_3Identityblock4_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  é
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
²

'__inference_model_2_layer_call_fn_10262
content_image
style_image
unknown
	unknown_0#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	%

unknown_31:@

unknown_32:@$

unknown_33:@@

unknown_34:@$

unknown_35:@

unknown_36:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallcontent_imagestyle_imageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_10101y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecontent_image:^Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namestyle_image: 

_output_shapes
:: 

_output_shapes
:
ó

G__inference_block1_conv1_layer_call_and_return_conditional_losses_11933

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
¤
,__inference_block3_conv2_layer_call_fn_12062

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8199
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

G
+__inference_block1_pool_layer_call_fn_11963

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_8129z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
÷
C__inference_conv2d_10_layer_call_and_return_conditional_losses_8827

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ÿ

G__inference_block4_conv1_layer_call_and_return_conditional_losses_12153

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
÷
C__inference_conv2d_11_layer_call_and_return_conditional_losses_8846

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¹b
â
A__inference_encoder_layer_call_and_return_conditional_losses_9504

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@®
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ì
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ï
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  x
IdentityIdentityblock1_conv1/Relu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{

Identity_1Identityblock2_conv1/Relu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿy

Identity_2Identityblock3_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@y

Identity_3Identityblock4_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  é
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
÷
C__inference_conv2d_9_layer_call_and_return_conditional_losses_12175

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

­
&__inference_decoder_layer_call_fn_9023
conv2d_9_input#
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	%

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@$

unknown_15:@

unknown_16:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_8984y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
(
_user_specified_nameconv2d_9_input
Ï
K
/__inference_up_sampling2d_3_layer_call_fn_12198

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8812i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ñ
Ê
__inference_call_12288

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ý
®

#__inference_signature_wrapper_10596
content_image
style_image
unknown
	unknown_0#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	%

unknown_31:@

unknown_32:@$

unknown_33:@@

unknown_34:@$

unknown_35:@

unknown_36:
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallcontent_imagestyle_imageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_8048y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecontent_image:^Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namestyle_image: 

_output_shapes
:: 

_output_shapes
:
ó

G__inference_block1_conv2_layer_call_and_return_conditional_losses_11953

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ó
K
/__inference_up_sampling2d_5_layer_call_fn_12468

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8944j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­g
ã
B__inference_encoder_layer_call_and_return_conditional_losses_11530

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Å
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Þ
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¶
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¾
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
IdentityIdentityblock1_conv1/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identityblock2_conv1/Relu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identityblock3_conv1/Relu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identityblock4_conv1/Relu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ºb
ã
B__inference_encoder_layer_call_and_return_conditional_losses_11676

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@®
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ì
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ï
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  x
IdentityIdentityblock1_conv1/Relu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{

Identity_1Identityblock2_conv1/Relu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿy

Identity_2Identityblock3_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@y

Identity_3Identityblock4_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  é
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó
K
/__inference_up_sampling2d_4_layer_call_fn_12368

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8897k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¹b
â
A__inference_encoder_layer_call_and_return_conditional_losses_9846

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@®
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ì
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ï
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿu
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  x
IdentityIdentityblock1_conv1/Relu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{

Identity_1Identityblock2_conv1/Relu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿy

Identity_2Identityblock3_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@y

Identity_3Identityblock4_conv1/Relu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  é
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
l
B__inference_ada_in_1_layer_call_and_return_conditional_losses_9594

inputs
inputs_1
identityo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      °
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeaninputs_1)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
moments_1/SquaredDifferenceSquaredDifferenceinputs_1moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ¶
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7r
addAddV2moments/variance:output:0add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
SqrtSqrtadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Sqrt_1Sqrt	add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
subSubinputsmoments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Z
mulMul
Sqrt_1:y:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
truedivRealDivmul:z:0Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  o
add_2AddV2truediv:z:0moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Z
IdentityIdentity	add_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

a
E__inference_block2_pool_layer_call_and_return_conditional_losses_8069

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

§
!__inference__traced_restore_12808
file_prefix>
$assignvariableop_block1_conv1_kernel:@2
$assignvariableop_1_block1_conv1_bias:@@
&assignvariableop_2_block1_conv2_kernel:@@2
$assignvariableop_3_block1_conv2_bias:@A
&assignvariableop_4_block2_conv1_kernel:@3
$assignvariableop_5_block2_conv1_bias:	B
&assignvariableop_6_block2_conv2_kernel:3
$assignvariableop_7_block2_conv2_bias:	B
&assignvariableop_8_block3_conv1_kernel:3
$assignvariableop_9_block3_conv1_bias:	C
'assignvariableop_10_block3_conv2_kernel:4
%assignvariableop_11_block3_conv2_bias:	C
'assignvariableop_12_block3_conv3_kernel:4
%assignvariableop_13_block3_conv3_bias:	C
'assignvariableop_14_block3_conv4_kernel:4
%assignvariableop_15_block3_conv4_bias:	C
'assignvariableop_16_block4_conv1_kernel:4
%assignvariableop_17_block4_conv1_bias:	?
#assignvariableop_18_conv2d_9_kernel:0
!assignvariableop_19_conv2d_9_bias:	@
$assignvariableop_20_conv2d_10_kernel:1
"assignvariableop_21_conv2d_10_bias:	@
$assignvariableop_22_conv2d_11_kernel:1
"assignvariableop_23_conv2d_11_bias:	@
$assignvariableop_24_conv2d_12_kernel:1
"assignvariableop_25_conv2d_12_bias:	@
$assignvariableop_26_conv2d_13_kernel:1
"assignvariableop_27_conv2d_13_bias:	@
$assignvariableop_28_conv2d_14_kernel:1
"assignvariableop_29_conv2d_14_bias:	?
$assignvariableop_30_conv2d_15_kernel:@0
"assignvariableop_31_conv2d_15_bias:@>
$assignvariableop_32_conv2d_16_kernel:@@0
"assignvariableop_33_conv2d_16_bias:@>
$assignvariableop_34_conv2d_17_kernel:@0
"assignvariableop_35_conv2d_17_bias:
identity_37¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ç
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*í
valueãBà%B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHº
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ª
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_12_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_12_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_13_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_13_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_15_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_15_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_16_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_16_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_17_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_17_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ç
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: Ô
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¢
±

&__inference_model_2_layer_call_fn_9717
content_image
style_image
unknown
	unknown_0#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	%

unknown_31:@

unknown_32:@$

unknown_33:@@

unknown_34:@$

unknown_35:@

unknown_36:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallcontent_imagestyle_imageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_9638y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecontent_image:^Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namestyle_image: 

_output_shapes
:: 

_output_shapes
:

ª

'__inference_model_2_layer_call_fn_10678
inputs_0
inputs_1
unknown
	unknown_0#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	%

unknown_31:@

unknown_32:@$

unknown_33:@@

unknown_34:@$

unknown_35:@

unknown_36:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_9638y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1: 

_output_shapes
:: 

_output_shapes
:
ó
¡
)__inference_conv2d_13_layer_call_fn_12332

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_8884x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

ö
D__inference_conv2d_15_layer_call_and_return_conditional_losses_12445

inputs9
conv2d_readvariableop_resource:@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
Æ
__inference_call_8037

inputs8
conv2d_readvariableop_resource:@)
add_readvariableop_resource:
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û
×
'__inference_encoder_layer_call_fn_11337

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9504y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿz

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@z

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
F__inference_block2_pool_layer_call_and_return_conditional_losses_12028

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
×
'__inference_encoder_layer_call_fn_11384

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9846y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿz

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@z

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

õ
C__inference_conv2d_15_layer_call_and_return_conditional_losses_8931

inputs9
conv2d_readvariableop_resource:@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
¤
,__inference_block4_conv1_layer_call_fn_12142

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8256
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_12133

inputs
identity
MaxPoolMaxPoolinputs*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
s
IdentityIdentityMaxPool:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ

F__inference_block3_conv4_layer_call_and_return_conditional_losses_8233

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
÷
C__inference_conv2d_12_layer_call_and_return_conditional_losses_8865

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
þ

F__inference_block2_conv2_layer_call_and_return_conditional_losses_8159

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ

F__inference_block3_conv2_layer_call_and_return_conditional_losses_8199

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
õ
D__inference_conv2d_17_layer_call_and_return_conditional_losses_12544

inputs8
conv2d_readvariableop_resource:@)
add_readvariableop_resource:
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð
É
__inference_call_7882

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

÷
C__inference_conv2d_14_layer_call_and_return_conditional_losses_8912

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0©
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0v
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿR
ReluReluadd:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

G__inference_block2_conv2_layer_call_and_return_conditional_losses_12013

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ

F__inference_block3_conv3_layer_call_and_return_conditional_losses_8216

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
o
C__inference_ada_in_1_layer_call_and_return_conditional_losses_11709
inputs_0
inputs_1
identityo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments/meanMeaninputs_0'moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(v
moments/StopGradientStopGradientmoments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
moments/SquaredDifferenceSquaredDifferenceinputs_0moments/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      °
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeaninputs_1)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
moments_1/SquaredDifferenceSquaredDifferenceinputs_1moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ¶
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7r
addAddV2moments/variance:output:0add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
SqrtSqrtadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Sqrt_1Sqrt	add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
subSubinputs_0moments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Z
mulMul
Sqrt_1:y:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
truedivRealDivmul:z:0Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  o
add_2AddV2truediv:z:0moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Z
IdentityIdentity	add_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1
þ
÷
C__inference_conv2d_13_layer_call_and_return_conditional_losses_8884

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

G
+__inference_block3_pool_layer_call_fn_12123

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_8243{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
É
__inference_call_7958

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ô

)__inference_conv2d_17_layer_call_fn_12532

inputs!
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_8977y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô

)__inference_conv2d_16_layer_call_fn_12497

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_8959y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ª

'__inference_model_2_layer_call_fn_10760
inputs_0
inputs_1
unknown
	unknown_0#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	%

unknown_31:@

unknown_32:@$

unknown_33:@@

unknown_34:@$

unknown_35:@

unknown_36:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%&'*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_10101y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1: 

_output_shapes
:: 

_output_shapes
:
Þ
e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8812

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"        X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(~
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ã
f
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_12488

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
ø
D__inference_conv2d_12_layer_call_and_return_conditional_losses_12310

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ï
¦
'__inference_decoder_layer_call_fn_11750

inputs#
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	%

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@$

unknown_15:@

unknown_16:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_8984y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ð
É
__inference_call_7922

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
í
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_8169

inputs
identity
MaxPoolMaxPoolinputs*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
s
IdentityIdentityMaxPool:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
F__inference_block1_pool_layer_call_and_return_conditional_losses_11968

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µG
á
__inference__traced_save_12690
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ä
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*í
valueãBà%B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH·
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B º
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ð
_input_shapesÞ
Û: :@:@:@@:@:@::::::::::::::::::::::::::@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:@:  

_output_shapes
:@:,!(
&
_output_shapes
:@@: "

_output_shapes
:@:,#(
&
_output_shapes
:@: $

_output_shapes
::%

_output_shapes
: 
¶9
¼
A__inference_decoder_layer_call_and_return_conditional_losses_8984

inputs)
conv2d_9_8800:
conv2d_9_8802:	*
conv2d_10_8828:
conv2d_10_8830:	*
conv2d_11_8847:
conv2d_11_8849:	*
conv2d_12_8866:
conv2d_12_8868:	*
conv2d_13_8885:
conv2d_13_8887:	*
conv2d_14_8913:
conv2d_14_8915:	)
conv2d_15_8932:@
conv2d_15_8934:@(
conv2d_16_8960:@@
conv2d_16_8962:@(
conv2d_17_8978:@
conv2d_17_8980:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallö
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_8800conv2d_9_8802*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_8799ó
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8812
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_10_8828conv2d_10_8830*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_8827
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_8847conv2d_11_8849*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_8846
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_8866conv2d_12_8868*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_8865
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_8885conv2d_13_8887*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_8884ö
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8897
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_8913conv2d_14_8915*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_8912
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_8932conv2d_15_8934*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_8931õ
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8944
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_16_8960conv2d_16_8962*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_8959
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_8978conv2d_17_8980*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_8977
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ò
Ç
__inference_call_12523

inputs8
conv2d_readvariableop_resource:@@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð
É
__inference_call_7904

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ó
¡
)__inference_conv2d_10_layer_call_fn_12227

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_8827x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ë@
®
B__inference_model_2_layer_call_and_return_conditional_losses_10101

inputs
inputs_1!
tf_nn_bias_add_4_biasadd_bias!
tf_nn_bias_add_3_biasadd_bias&
encoder_9996:@
encoder_9998:@'
encoder_10000:@@
encoder_10002:@(
encoder_10004:@
encoder_10006:	)
encoder_10008:
encoder_10010:	)
encoder_10012:
encoder_10014:	)
encoder_10016:
encoder_10018:	)
encoder_10020:
encoder_10022:	)
encoder_10024:
encoder_10026:	)
encoder_10028:
encoder_10030:	)
decoder_10059:
decoder_10061:	)
decoder_10063:
decoder_10065:	)
decoder_10067:
decoder_10069:	)
decoder_10071:
decoder_10073:	)
decoder_10075:
decoder_10077:	)
decoder_10079:
decoder_10081:	(
decoder_10083:@
decoder_10085:@'
decoder_10087:@@
decoder_10089:@'
decoder_10091:@
decoder_10093:
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall¢!encoder/StatefulPartitionedCall_1n
tf.reverse_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_4/ReverseV2	ReverseV2inputs_1$tf.reverse_4/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
tf.reverse_3/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_3/ReverseV2	ReverseV2inputs$tf.reverse_3/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_4/strided_sliceStridedSlicetf.reverse_4/ReverseV2:output:07tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_3/strided_sliceStridedSlicetf.reverse_3/ReverseV2:output:07tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask±
tf.nn.bias_add_4/BiasAddBiasAdd1tf.__operators__.getitem_4/strided_slice:output:0tf_nn_bias_add_4_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
encoder/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0encoder_9996encoder_9998encoder_10000encoder_10002encoder_10004encoder_10006encoder_10008encoder_10010encoder_10012encoder_10014encoder_10016encoder_10018encoder_10020encoder_10022encoder_10024encoder_10026encoder_10028encoder_10030*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9846ú
!encoder/StatefulPartitionedCall_1StatefulPartitionedCall!tf.nn.bias_add_4/BiasAdd:output:0encoder_9996encoder_9998encoder_10000encoder_10002encoder_10004encoder_10006encoder_10008encoder_10010encoder_10012encoder_10014encoder_10016encoder_10018encoder_10020encoder_10022encoder_10024encoder_10026encoder_10028encoder_10030*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9846
ada_in_1/PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:3*encoder/StatefulPartitionedCall_1:output:3*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_ada_in_1_layer_call_and_return_conditional_losses_9594 
decoder/StatefulPartitionedCallStatefulPartitionedCall!ada_in_1/PartitionedCall:output:0decoder_10059decoder_10061decoder_10063decoder_10065decoder_10067decoder_10069decoder_10071decoder_10073decoder_10075decoder_10077decoder_10079decoder_10081decoder_10083decoder_10085decoder_10087decoder_10089decoder_10091decoder_10093*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_9223o
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÎ
(tf.clip_by_value_1/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Â
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity$tf.clip_by_value_1/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall"^encoder/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2F
!encoder/StatefulPartitionedCall_1!encoder/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
»
¡
,__inference_block1_conv2_layer_call_fn_11942

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_8119
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ß
f
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_12218

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"        X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(~
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ó
¡
)__inference_conv2d_11_layer_call_fn_12262

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_8846x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
°
G
+__inference_block1_pool_layer_call_fn_11958

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_8057
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö¼
Æ"
B__inference_model_2_layer_call_and_return_conditional_losses_10978
inputs_0
inputs_1!
tf_nn_bias_add_4_biasadd_bias!
tf_nn_bias_add_3_biasadd_biasM
3encoder_block1_conv1_conv2d_readvariableop_resource:@B
4encoder_block1_conv1_biasadd_readvariableop_resource:@M
3encoder_block1_conv2_conv2d_readvariableop_resource:@@B
4encoder_block1_conv2_biasadd_readvariableop_resource:@N
3encoder_block2_conv1_conv2d_readvariableop_resource:@C
4encoder_block2_conv1_biasadd_readvariableop_resource:	O
3encoder_block2_conv2_conv2d_readvariableop_resource:C
4encoder_block2_conv2_biasadd_readvariableop_resource:	O
3encoder_block3_conv1_conv2d_readvariableop_resource:C
4encoder_block3_conv1_biasadd_readvariableop_resource:	O
3encoder_block3_conv2_conv2d_readvariableop_resource:C
4encoder_block3_conv2_biasadd_readvariableop_resource:	O
3encoder_block3_conv3_conv2d_readvariableop_resource:C
4encoder_block3_conv3_biasadd_readvariableop_resource:	O
3encoder_block3_conv4_conv2d_readvariableop_resource:C
4encoder_block3_conv4_biasadd_readvariableop_resource:	O
3encoder_block4_conv1_conv2d_readvariableop_resource:C
4encoder_block4_conv1_biasadd_readvariableop_resource:	2
decoder_conv2d_9_10916:%
decoder_conv2d_9_10918:	3
decoder_conv2d_10_10925:&
decoder_conv2d_10_10927:	3
decoder_conv2d_11_10930:&
decoder_conv2d_11_10932:	3
decoder_conv2d_12_10935:&
decoder_conv2d_12_10937:	3
decoder_conv2d_13_10940:&
decoder_conv2d_13_10942:	3
decoder_conv2d_14_10949:&
decoder_conv2d_14_10951:	2
decoder_conv2d_15_10954:@%
decoder_conv2d_15_10956:@1
decoder_conv2d_16_10963:@@%
decoder_conv2d_16_10965:@1
decoder_conv2d_17_10968:@%
decoder_conv2d_17_10970:
identity¢)decoder/conv2d_10/StatefulPartitionedCall¢)decoder/conv2d_11/StatefulPartitionedCall¢)decoder/conv2d_12/StatefulPartitionedCall¢)decoder/conv2d_13/StatefulPartitionedCall¢)decoder/conv2d_14/StatefulPartitionedCall¢)decoder/conv2d_15/StatefulPartitionedCall¢)decoder/conv2d_16/StatefulPartitionedCall¢)decoder/conv2d_17/StatefulPartitionedCall¢(decoder/conv2d_9/StatefulPartitionedCall¢+encoder/block1_conv1/BiasAdd/ReadVariableOp¢-encoder/block1_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block1_conv1/Conv2D/ReadVariableOp¢,encoder/block1_conv1/Conv2D_1/ReadVariableOp¢+encoder/block1_conv2/BiasAdd/ReadVariableOp¢-encoder/block1_conv2/BiasAdd_1/ReadVariableOp¢*encoder/block1_conv2/Conv2D/ReadVariableOp¢,encoder/block1_conv2/Conv2D_1/ReadVariableOp¢+encoder/block2_conv1/BiasAdd/ReadVariableOp¢-encoder/block2_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block2_conv1/Conv2D/ReadVariableOp¢,encoder/block2_conv1/Conv2D_1/ReadVariableOp¢+encoder/block2_conv2/BiasAdd/ReadVariableOp¢-encoder/block2_conv2/BiasAdd_1/ReadVariableOp¢*encoder/block2_conv2/Conv2D/ReadVariableOp¢,encoder/block2_conv2/Conv2D_1/ReadVariableOp¢+encoder/block3_conv1/BiasAdd/ReadVariableOp¢-encoder/block3_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv1/Conv2D/ReadVariableOp¢,encoder/block3_conv1/Conv2D_1/ReadVariableOp¢+encoder/block3_conv2/BiasAdd/ReadVariableOp¢-encoder/block3_conv2/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv2/Conv2D/ReadVariableOp¢,encoder/block3_conv2/Conv2D_1/ReadVariableOp¢+encoder/block3_conv3/BiasAdd/ReadVariableOp¢-encoder/block3_conv3/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv3/Conv2D/ReadVariableOp¢,encoder/block3_conv3/Conv2D_1/ReadVariableOp¢+encoder/block3_conv4/BiasAdd/ReadVariableOp¢-encoder/block3_conv4/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv4/Conv2D/ReadVariableOp¢,encoder/block3_conv4/Conv2D_1/ReadVariableOp¢+encoder/block4_conv1/BiasAdd/ReadVariableOp¢-encoder/block4_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block4_conv1/Conv2D/ReadVariableOp¢,encoder/block4_conv1/Conv2D_1/ReadVariableOpn
tf.reverse_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_4/ReverseV2	ReverseV2inputs_1$tf.reverse_4/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
tf.reverse_3/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_3/ReverseV2	ReverseV2inputs_0$tf.reverse_3/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_4/strided_sliceStridedSlicetf.reverse_4/ReverseV2:output:07tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_3/strided_sliceStridedSlicetf.reverse_3/ReverseV2:output:07tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask±
tf.nn.bias_add_4/BiasAddBiasAdd1tf.__operators__.getitem_4/strided_slice:output:0tf_nn_bias_add_4_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
*encoder/block1_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0à
encoder/block1_conv1/Conv2DConv2D!tf.nn.bias_add_3/BiasAdd:output:02encoder/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+encoder/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¾
encoder/block1_conv1/BiasAddBiasAdd$encoder/block1_conv1/Conv2D:output:03encoder/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv1/ReluRelu%encoder/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
*encoder/block1_conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0æ
encoder/block1_conv2/Conv2DConv2D'encoder/block1_conv1/Relu:activations:02encoder/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+encoder/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¾
encoder/block1_conv2/BiasAddBiasAdd$encoder/block1_conv2/Conv2D:output:03encoder/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv2/ReluRelu%encoder/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
encoder/block1_pool/MaxPoolMaxPool'encoder/block1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
§
*encoder/block2_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ä
encoder/block2_conv1/Conv2DConv2D$encoder/block1_pool/MaxPool:output:02encoder/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+encoder/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¿
encoder/block2_conv1/BiasAddBiasAdd$encoder/block2_conv1/Conv2D:output:03encoder/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv1/ReluRelu%encoder/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ¨
*encoder/block2_conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ç
encoder/block2_conv2/Conv2DConv2D'encoder/block2_conv1/Relu:activations:02encoder/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+encoder/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¿
encoder/block2_conv2/BiasAddBiasAdd$encoder/block2_conv2/Conv2D:output:03encoder/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv2/ReluRelu%encoder/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ½
encoder/block2_pool/MaxPoolMaxPool'encoder/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides
¨
*encoder/block3_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0â
encoder/block3_conv1/Conv2DConv2D$encoder/block2_pool/MaxPool:output:02encoder/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv1/BiasAddBiasAdd$encoder/block3_conv1/Conv2D:output:03encoder/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv1/ReluRelu%encoder/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¨
*encoder/block3_conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
encoder/block3_conv2/Conv2DConv2D'encoder/block3_conv1/Relu:activations:02encoder/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv2/BiasAddBiasAdd$encoder/block3_conv2/Conv2D:output:03encoder/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv2/ReluRelu%encoder/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¨
*encoder/block3_conv3/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
encoder/block3_conv3/Conv2DConv2D'encoder/block3_conv2/Relu:activations:02encoder/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv3/BiasAddBiasAdd$encoder/block3_conv3/Conv2D:output:03encoder/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv3/ReluRelu%encoder/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¨
*encoder/block3_conv4/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
encoder/block3_conv4/Conv2DConv2D'encoder/block3_conv3/Relu:activations:02encoder/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv4/BiasAddBiasAdd$encoder/block3_conv4/Conv2D:output:03encoder/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv4/ReluRelu%encoder/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@½
encoder/block3_pool/MaxPoolMaxPool'encoder/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
¨
*encoder/block4_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0â
encoder/block4_conv1/Conv2DConv2D$encoder/block3_pool/MaxPool:output:02encoder/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

+encoder/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block4_conv1/BiasAddBiasAdd$encoder/block4_conv1/Conv2D:output:03encoder/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
encoder/block4_conv1/ReluRelu%encoder/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¨
,encoder/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ä
encoder/block1_conv1/Conv2D_1Conv2D!tf.nn.bias_add_4/BiasAdd:output:04encoder/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

-encoder/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ä
encoder/block1_conv1/BiasAdd_1BiasAdd&encoder/block1_conv1/Conv2D_1:output:05encoder/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv1/Relu_1Relu'encoder/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
,encoder/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ì
encoder/block1_conv2/Conv2D_1Conv2D)encoder/block1_conv1/Relu_1:activations:04encoder/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

-encoder/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ä
encoder/block1_conv2/BiasAdd_1BiasAdd&encoder/block1_conv2/Conv2D_1:output:05encoder/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv2/Relu_1Relu'encoder/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
encoder/block1_pool/MaxPool_1MaxPool)encoder/block1_conv2/Relu_1:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
©
,encoder/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ê
encoder/block2_conv1/Conv2D_1Conv2D&encoder/block1_pool/MaxPool_1:output:04encoder/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

-encoder/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Å
encoder/block2_conv1/BiasAdd_1BiasAdd&encoder/block2_conv1/Conv2D_1:output:05encoder/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv1/Relu_1Relu'encoder/block2_conv1/BiasAdd_1:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿª
,encoder/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0í
encoder/block2_conv2/Conv2D_1Conv2D)encoder/block2_conv1/Relu_1:activations:04encoder/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

-encoder/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Å
encoder/block2_conv2/BiasAdd_1BiasAdd&encoder/block2_conv2/Conv2D_1:output:05encoder/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv2/Relu_1Relu'encoder/block2_conv2/BiasAdd_1:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÁ
encoder/block2_pool/MaxPool_1MaxPool)encoder/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides
ª
,encoder/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0è
encoder/block3_conv1/Conv2D_1Conv2D&encoder/block2_pool/MaxPool_1:output:04encoder/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv1/BiasAdd_1BiasAdd&encoder/block3_conv1/Conv2D_1:output:05encoder/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv1/Relu_1Relu'encoder/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ª
,encoder/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
encoder/block3_conv2/Conv2D_1Conv2D)encoder/block3_conv1/Relu_1:activations:04encoder/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv2/BiasAdd_1BiasAdd&encoder/block3_conv2/Conv2D_1:output:05encoder/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv2/Relu_1Relu'encoder/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ª
,encoder/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
encoder/block3_conv3/Conv2D_1Conv2D)encoder/block3_conv2/Relu_1:activations:04encoder/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv3/BiasAdd_1BiasAdd&encoder/block3_conv3/Conv2D_1:output:05encoder/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv3/Relu_1Relu'encoder/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ª
,encoder/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
encoder/block3_conv4/Conv2D_1Conv2D)encoder/block3_conv3/Relu_1:activations:04encoder/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv4/BiasAdd_1BiasAdd&encoder/block3_conv4/Conv2D_1:output:05encoder/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv4/Relu_1Relu'encoder/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@Á
encoder/block3_pool/MaxPool_1MaxPool)encoder/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
ª
,encoder/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0è
encoder/block4_conv1/Conv2D_1Conv2D&encoder/block3_pool/MaxPool_1:output:04encoder/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

-encoder/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block4_conv1/BiasAdd_1BiasAdd&encoder/block4_conv1/Conv2D_1:output:05encoder/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
encoder/block4_conv1/Relu_1Relu'encoder/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  x
'ada_in_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ä
ada_in_1/moments/meanMean'encoder/block4_conv1/Relu:activations:00ada_in_1/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
ada_in_1/moments/StopGradientStopGradientada_in_1/moments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
"ada_in_1/moments/SquaredDifferenceSquaredDifference'encoder/block4_conv1/Relu:activations:0&ada_in_1/moments/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  |
+ada_in_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ë
ada_in_1/moments/varianceMean&ada_in_1/moments/SquaredDifference:z:04ada_in_1/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(z
)ada_in_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ê
ada_in_1/moments_1/meanMean)encoder/block4_conv1/Relu_1:activations:02ada_in_1/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
ada_in_1/moments_1/StopGradientStopGradient ada_in_1/moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
$ada_in_1/moments_1/SquaredDifferenceSquaredDifference)encoder/block4_conv1/Relu_1:activations:0(ada_in_1/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ~
-ada_in_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ñ
ada_in_1/moments_1/varianceMean(ada_in_1/moments_1/SquaredDifference:z:06ada_in_1/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(S
ada_in_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
ada_in_1/addAddV2"ada_in_1/moments/variance:output:0ada_in_1/add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
ada_in_1/SqrtSqrtada_in_1/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
ada_in_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
ada_in_1/add_1AddV2$ada_in_1/moments_1/variance:output:0ada_in_1/add_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
ada_in_1/Sqrt_1Sqrtada_in_1/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ada_in_1/subSub'encoder/block4_conv1/Relu:activations:0ada_in_1/moments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  u
ada_in_1/mulMulada_in_1/Sqrt_1:y:0ada_in_1/sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  {
ada_in_1/truedivRealDivada_in_1/mul:z:0ada_in_1/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
ada_in_1/add_2AddV2ada_in_1/truediv:z:0 ada_in_1/moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ï
(decoder/conv2d_9/StatefulPartitionedCallStatefulPartitionedCallada_in_1/add_2:z:0decoder_conv2d_9_10916decoder_conv2d_9_10918*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7882n
decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"        p
decoder/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d_3/mulMul&decoder/up_sampling2d_3/Const:output:0(decoder/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:ö
4decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor1decoder/conv2d_9/StatefulPartitionedCall:output:0decoder/up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(¥
)decoder/conv2d_10/StatefulPartitionedCallStatefulPartitionedCallEdecoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0decoder_conv2d_10_10925decoder_conv2d_10_10927*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7904
)decoder/conv2d_11/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_10/StatefulPartitionedCall:output:0decoder_conv2d_11_10930decoder_conv2d_11_10932*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7922
)decoder/conv2d_12/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_11/StatefulPartitionedCall:output:0decoder_conv2d_12_10935decoder_conv2d_12_10937*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7940
)decoder/conv2d_13/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_12/StatefulPartitionedCall:output:0decoder_conv2d_13_10940decoder_conv2d_13_10942*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7958n
decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   p
decoder/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d_4/mulMul&decoder/up_sampling2d_4/Const:output:0(decoder/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:ù
4decoder/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor2decoder/conv2d_13/StatefulPartitionedCall:output:0decoder/up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(§
)decoder/conv2d_14/StatefulPartitionedCallStatefulPartitionedCallEdecoder/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0decoder_conv2d_14_10949decoder_conv2d_14_10951*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7980
)decoder/conv2d_15/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_14/StatefulPartitionedCall:output:0decoder_conv2d_15_10954decoder_conv2d_15_10956*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7998n
decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d_5/mulMul&decoder/up_sampling2d_5/Const:output:0(decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:ø
4decoder/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor2decoder/conv2d_15/StatefulPartitionedCall:output:0decoder/up_sampling2d_5/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(¦
)decoder/conv2d_16/StatefulPartitionedCallStatefulPartitionedCallEdecoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0decoder_conv2d_16_10963decoder_conv2d_16_10965*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8020
)decoder/conv2d_17/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_16/StatefulPartitionedCall:output:0decoder_conv2d_17_10968decoder_conv2d_17_10970*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8037o
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CØ
(tf.clip_by_value_1/clip_by_value/MinimumMinimum2decoder/conv2d_17/StatefulPartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Â
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity$tf.clip_by_value_1/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
NoOpNoOp*^decoder/conv2d_10/StatefulPartitionedCall*^decoder/conv2d_11/StatefulPartitionedCall*^decoder/conv2d_12/StatefulPartitionedCall*^decoder/conv2d_13/StatefulPartitionedCall*^decoder/conv2d_14/StatefulPartitionedCall*^decoder/conv2d_15/StatefulPartitionedCall*^decoder/conv2d_16/StatefulPartitionedCall*^decoder/conv2d_17/StatefulPartitionedCall)^decoder/conv2d_9/StatefulPartitionedCall,^encoder/block1_conv1/BiasAdd/ReadVariableOp.^encoder/block1_conv1/BiasAdd_1/ReadVariableOp+^encoder/block1_conv1/Conv2D/ReadVariableOp-^encoder/block1_conv1/Conv2D_1/ReadVariableOp,^encoder/block1_conv2/BiasAdd/ReadVariableOp.^encoder/block1_conv2/BiasAdd_1/ReadVariableOp+^encoder/block1_conv2/Conv2D/ReadVariableOp-^encoder/block1_conv2/Conv2D_1/ReadVariableOp,^encoder/block2_conv1/BiasAdd/ReadVariableOp.^encoder/block2_conv1/BiasAdd_1/ReadVariableOp+^encoder/block2_conv1/Conv2D/ReadVariableOp-^encoder/block2_conv1/Conv2D_1/ReadVariableOp,^encoder/block2_conv2/BiasAdd/ReadVariableOp.^encoder/block2_conv2/BiasAdd_1/ReadVariableOp+^encoder/block2_conv2/Conv2D/ReadVariableOp-^encoder/block2_conv2/Conv2D_1/ReadVariableOp,^encoder/block3_conv1/BiasAdd/ReadVariableOp.^encoder/block3_conv1/BiasAdd_1/ReadVariableOp+^encoder/block3_conv1/Conv2D/ReadVariableOp-^encoder/block3_conv1/Conv2D_1/ReadVariableOp,^encoder/block3_conv2/BiasAdd/ReadVariableOp.^encoder/block3_conv2/BiasAdd_1/ReadVariableOp+^encoder/block3_conv2/Conv2D/ReadVariableOp-^encoder/block3_conv2/Conv2D_1/ReadVariableOp,^encoder/block3_conv3/BiasAdd/ReadVariableOp.^encoder/block3_conv3/BiasAdd_1/ReadVariableOp+^encoder/block3_conv3/Conv2D/ReadVariableOp-^encoder/block3_conv3/Conv2D_1/ReadVariableOp,^encoder/block3_conv4/BiasAdd/ReadVariableOp.^encoder/block3_conv4/BiasAdd_1/ReadVariableOp+^encoder/block3_conv4/Conv2D/ReadVariableOp-^encoder/block3_conv4/Conv2D_1/ReadVariableOp,^encoder/block4_conv1/BiasAdd/ReadVariableOp.^encoder/block4_conv1/BiasAdd_1/ReadVariableOp+^encoder/block4_conv1/Conv2D/ReadVariableOp-^encoder/block4_conv1/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)decoder/conv2d_10/StatefulPartitionedCall)decoder/conv2d_10/StatefulPartitionedCall2V
)decoder/conv2d_11/StatefulPartitionedCall)decoder/conv2d_11/StatefulPartitionedCall2V
)decoder/conv2d_12/StatefulPartitionedCall)decoder/conv2d_12/StatefulPartitionedCall2V
)decoder/conv2d_13/StatefulPartitionedCall)decoder/conv2d_13/StatefulPartitionedCall2V
)decoder/conv2d_14/StatefulPartitionedCall)decoder/conv2d_14/StatefulPartitionedCall2V
)decoder/conv2d_15/StatefulPartitionedCall)decoder/conv2d_15/StatefulPartitionedCall2V
)decoder/conv2d_16/StatefulPartitionedCall)decoder/conv2d_16/StatefulPartitionedCall2V
)decoder/conv2d_17/StatefulPartitionedCall)decoder/conv2d_17/StatefulPartitionedCall2T
(decoder/conv2d_9/StatefulPartitionedCall(decoder/conv2d_9/StatefulPartitionedCall2Z
+encoder/block1_conv1/BiasAdd/ReadVariableOp+encoder/block1_conv1/BiasAdd/ReadVariableOp2^
-encoder/block1_conv1/BiasAdd_1/ReadVariableOp-encoder/block1_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block1_conv1/Conv2D/ReadVariableOp*encoder/block1_conv1/Conv2D/ReadVariableOp2\
,encoder/block1_conv1/Conv2D_1/ReadVariableOp,encoder/block1_conv1/Conv2D_1/ReadVariableOp2Z
+encoder/block1_conv2/BiasAdd/ReadVariableOp+encoder/block1_conv2/BiasAdd/ReadVariableOp2^
-encoder/block1_conv2/BiasAdd_1/ReadVariableOp-encoder/block1_conv2/BiasAdd_1/ReadVariableOp2X
*encoder/block1_conv2/Conv2D/ReadVariableOp*encoder/block1_conv2/Conv2D/ReadVariableOp2\
,encoder/block1_conv2/Conv2D_1/ReadVariableOp,encoder/block1_conv2/Conv2D_1/ReadVariableOp2Z
+encoder/block2_conv1/BiasAdd/ReadVariableOp+encoder/block2_conv1/BiasAdd/ReadVariableOp2^
-encoder/block2_conv1/BiasAdd_1/ReadVariableOp-encoder/block2_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block2_conv1/Conv2D/ReadVariableOp*encoder/block2_conv1/Conv2D/ReadVariableOp2\
,encoder/block2_conv1/Conv2D_1/ReadVariableOp,encoder/block2_conv1/Conv2D_1/ReadVariableOp2Z
+encoder/block2_conv2/BiasAdd/ReadVariableOp+encoder/block2_conv2/BiasAdd/ReadVariableOp2^
-encoder/block2_conv2/BiasAdd_1/ReadVariableOp-encoder/block2_conv2/BiasAdd_1/ReadVariableOp2X
*encoder/block2_conv2/Conv2D/ReadVariableOp*encoder/block2_conv2/Conv2D/ReadVariableOp2\
,encoder/block2_conv2/Conv2D_1/ReadVariableOp,encoder/block2_conv2/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv1/BiasAdd/ReadVariableOp+encoder/block3_conv1/BiasAdd/ReadVariableOp2^
-encoder/block3_conv1/BiasAdd_1/ReadVariableOp-encoder/block3_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv1/Conv2D/ReadVariableOp*encoder/block3_conv1/Conv2D/ReadVariableOp2\
,encoder/block3_conv1/Conv2D_1/ReadVariableOp,encoder/block3_conv1/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv2/BiasAdd/ReadVariableOp+encoder/block3_conv2/BiasAdd/ReadVariableOp2^
-encoder/block3_conv2/BiasAdd_1/ReadVariableOp-encoder/block3_conv2/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv2/Conv2D/ReadVariableOp*encoder/block3_conv2/Conv2D/ReadVariableOp2\
,encoder/block3_conv2/Conv2D_1/ReadVariableOp,encoder/block3_conv2/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv3/BiasAdd/ReadVariableOp+encoder/block3_conv3/BiasAdd/ReadVariableOp2^
-encoder/block3_conv3/BiasAdd_1/ReadVariableOp-encoder/block3_conv3/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv3/Conv2D/ReadVariableOp*encoder/block3_conv3/Conv2D/ReadVariableOp2\
,encoder/block3_conv3/Conv2D_1/ReadVariableOp,encoder/block3_conv3/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv4/BiasAdd/ReadVariableOp+encoder/block3_conv4/BiasAdd/ReadVariableOp2^
-encoder/block3_conv4/BiasAdd_1/ReadVariableOp-encoder/block3_conv4/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv4/Conv2D/ReadVariableOp*encoder/block3_conv4/Conv2D/ReadVariableOp2\
,encoder/block3_conv4/Conv2D_1/ReadVariableOp,encoder/block3_conv4/Conv2D_1/ReadVariableOp2Z
+encoder/block4_conv1/BiasAdd/ReadVariableOp+encoder/block4_conv1/BiasAdd/ReadVariableOp2^
-encoder/block4_conv1/BiasAdd_1/ReadVariableOp-encoder/block4_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block4_conv1/Conv2D/ReadVariableOp*encoder/block4_conv1/Conv2D/ReadVariableOp2\
,encoder/block4_conv1/Conv2D_1/ReadVariableOp,encoder/block4_conv1/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1: 

_output_shapes
:: 

_output_shapes
:

e
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8776

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
K
/__inference_up_sampling2d_4_layer_call_fn_12363

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8757
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
¡
)__inference_conv2d_14_layer_call_fn_12397

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_8912z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

G__inference_block3_conv2_layer_call_and_return_conditional_losses_12073

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
ø
D__inference_conv2d_13_layer_call_and_return_conditional_losses_12345

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ò
ÿ
F__inference_block1_conv2_layer_call_and_return_conditional_losses_8119

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î9
Ä
A__inference_decoder_layer_call_and_return_conditional_losses_9355
conv2d_9_input)
conv2d_9_9306:
conv2d_9_9308:	*
conv2d_10_9312:
conv2d_10_9314:	*
conv2d_11_9317:
conv2d_11_9319:	*
conv2d_12_9322:
conv2d_12_9324:	*
conv2d_13_9327:
conv2d_13_9329:	*
conv2d_14_9333:
conv2d_14_9335:	)
conv2d_15_9338:@
conv2d_15_9340:@(
conv2d_16_9344:@@
conv2d_16_9346:@(
conv2d_17_9349:@
conv2d_17_9351:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallþ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputconv2d_9_9306conv2d_9_9308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_8799ó
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8812
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_10_9312conv2d_10_9314*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_8827
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_9317conv2d_11_9319*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_8846
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_9322conv2d_12_9324*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_8865
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_9327conv2d_13_9329*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_8884ö
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8897
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_9333conv2d_14_9335*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_8912
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_9338conv2d_15_9340*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_8931õ
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8944
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_16_9344conv2d_16_9346*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_8959
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_9349conv2d_17_9351*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_8977
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
(
_user_specified_nameconv2d_9_input
¸
K
/__inference_up_sampling2d_3_layer_call_fn_12193

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8738
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
¤
,__inference_block2_conv2_layer_call_fn_12002

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8159
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

×
&__inference_encoder_layer_call_fn_8311
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *Í
_output_shapesº
·:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_8266
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
þ

F__inference_block3_conv1_layer_call_and_return_conditional_losses_8182

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_8243

inputs
identity
MaxPoolMaxPoolinputs*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
s
IdentityIdentityMaxPool:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Æ
__inference_call_8020

inputs8
conv2d_readvariableop_resource:@@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
É
__inference_call_7980

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0©
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0v
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿR
ReluReluadd:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³B
Á	
A__inference_encoder_layer_call_and_return_conditional_losses_8266

inputs+
block1_conv1_8103:@
block1_conv1_8105:@+
block1_conv2_8120:@@
block1_conv2_8122:@,
block2_conv1_8143:@ 
block2_conv1_8145:	-
block2_conv2_8160: 
block2_conv2_8162:	-
block3_conv1_8183: 
block3_conv1_8185:	-
block3_conv2_8200: 
block3_conv2_8202:	-
block3_conv3_8217: 
block3_conv3_8219:	-
block3_conv4_8234: 
block3_conv4_8236:	-
block4_conv1_8257: 
block4_conv1_8259:	
identity

identity_1

identity_2

identity_3¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_8103block1_conv1_8105*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_8102¾
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8120block1_conv2_8122*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_8119
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_8129¶
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_8143block2_conv1_8145*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8142¿
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8160block2_conv2_8162*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8159
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_8169¶
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_8183block3_conv1_8185*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8182¿
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8200block3_conv2_8202*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8199¿
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8217block3_conv3_8219*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8216¿
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8234block3_conv4_8236*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_8233
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_8243¶
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_8257block4_conv1_8259*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8256
IdentityIdentity-block1_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity-block2_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity-block3_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity-block4_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
ÿ
F__inference_block1_conv1_layer_call_and_return_conditional_losses_8102

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À;
Ï
B__inference_decoder_layer_call_and_return_conditional_losses_11913

inputs*
conv2d_9_11855:
conv2d_9_11857:	+
conv2d_10_11864:
conv2d_10_11866:	+
conv2d_11_11869:
conv2d_11_11871:	+
conv2d_12_11874:
conv2d_12_11876:	+
conv2d_13_11879:
conv2d_13_11881:	+
conv2d_14_11888:
conv2d_14_11890:	*
conv2d_15_11893:@
conv2d_15_11895:@)
conv2d_16_11902:@@
conv2d_16_11904:@)
conv2d_17_11907:@
conv2d_17_11909:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallË
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_11855conv2d_9_11857*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7882f
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:Þ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor)conv2d_9/StatefulPartitionedCall:output:0up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0conv2d_10_11864conv2d_10_11866*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7904ò
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_11869conv2d_11_11871*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7922ò
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_11874conv2d_12_11876*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7940ò
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_11879conv2d_13_11881*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7958f
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   h
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:á
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor*conv2d_13/StatefulPartitionedCall:output:0up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0conv2d_14_11888conv2d_14_11890*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7980ó
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_11893conv2d_15_11895*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7998f
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:à
,up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor*conv2d_15/StatefulPartitionedCall:output:0up_sampling2d_5/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0conv2d_16_11902conv2d_16_11904*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8020ó
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_11907conv2d_17_11909*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8037
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
÷

)__inference_conv2d_15_layer_call_fn_12432

inputs"
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_8931y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_12210

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
Ê
__inference_call_12423

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0©
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0v
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿR
ReluReluadd:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú

F__inference_block2_conv1_layer_call_and_return_conditional_losses_8142

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ö¼
Æ"
B__inference_model_2_layer_call_and_return_conditional_losses_11196
inputs_0
inputs_1!
tf_nn_bias_add_4_biasadd_bias!
tf_nn_bias_add_3_biasadd_biasM
3encoder_block1_conv1_conv2d_readvariableop_resource:@B
4encoder_block1_conv1_biasadd_readvariableop_resource:@M
3encoder_block1_conv2_conv2d_readvariableop_resource:@@B
4encoder_block1_conv2_biasadd_readvariableop_resource:@N
3encoder_block2_conv1_conv2d_readvariableop_resource:@C
4encoder_block2_conv1_biasadd_readvariableop_resource:	O
3encoder_block2_conv2_conv2d_readvariableop_resource:C
4encoder_block2_conv2_biasadd_readvariableop_resource:	O
3encoder_block3_conv1_conv2d_readvariableop_resource:C
4encoder_block3_conv1_biasadd_readvariableop_resource:	O
3encoder_block3_conv2_conv2d_readvariableop_resource:C
4encoder_block3_conv2_biasadd_readvariableop_resource:	O
3encoder_block3_conv3_conv2d_readvariableop_resource:C
4encoder_block3_conv3_biasadd_readvariableop_resource:	O
3encoder_block3_conv4_conv2d_readvariableop_resource:C
4encoder_block3_conv4_biasadd_readvariableop_resource:	O
3encoder_block4_conv1_conv2d_readvariableop_resource:C
4encoder_block4_conv1_biasadd_readvariableop_resource:	2
decoder_conv2d_9_11134:%
decoder_conv2d_9_11136:	3
decoder_conv2d_10_11143:&
decoder_conv2d_10_11145:	3
decoder_conv2d_11_11148:&
decoder_conv2d_11_11150:	3
decoder_conv2d_12_11153:&
decoder_conv2d_12_11155:	3
decoder_conv2d_13_11158:&
decoder_conv2d_13_11160:	3
decoder_conv2d_14_11167:&
decoder_conv2d_14_11169:	2
decoder_conv2d_15_11172:@%
decoder_conv2d_15_11174:@1
decoder_conv2d_16_11181:@@%
decoder_conv2d_16_11183:@1
decoder_conv2d_17_11186:@%
decoder_conv2d_17_11188:
identity¢)decoder/conv2d_10/StatefulPartitionedCall¢)decoder/conv2d_11/StatefulPartitionedCall¢)decoder/conv2d_12/StatefulPartitionedCall¢)decoder/conv2d_13/StatefulPartitionedCall¢)decoder/conv2d_14/StatefulPartitionedCall¢)decoder/conv2d_15/StatefulPartitionedCall¢)decoder/conv2d_16/StatefulPartitionedCall¢)decoder/conv2d_17/StatefulPartitionedCall¢(decoder/conv2d_9/StatefulPartitionedCall¢+encoder/block1_conv1/BiasAdd/ReadVariableOp¢-encoder/block1_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block1_conv1/Conv2D/ReadVariableOp¢,encoder/block1_conv1/Conv2D_1/ReadVariableOp¢+encoder/block1_conv2/BiasAdd/ReadVariableOp¢-encoder/block1_conv2/BiasAdd_1/ReadVariableOp¢*encoder/block1_conv2/Conv2D/ReadVariableOp¢,encoder/block1_conv2/Conv2D_1/ReadVariableOp¢+encoder/block2_conv1/BiasAdd/ReadVariableOp¢-encoder/block2_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block2_conv1/Conv2D/ReadVariableOp¢,encoder/block2_conv1/Conv2D_1/ReadVariableOp¢+encoder/block2_conv2/BiasAdd/ReadVariableOp¢-encoder/block2_conv2/BiasAdd_1/ReadVariableOp¢*encoder/block2_conv2/Conv2D/ReadVariableOp¢,encoder/block2_conv2/Conv2D_1/ReadVariableOp¢+encoder/block3_conv1/BiasAdd/ReadVariableOp¢-encoder/block3_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv1/Conv2D/ReadVariableOp¢,encoder/block3_conv1/Conv2D_1/ReadVariableOp¢+encoder/block3_conv2/BiasAdd/ReadVariableOp¢-encoder/block3_conv2/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv2/Conv2D/ReadVariableOp¢,encoder/block3_conv2/Conv2D_1/ReadVariableOp¢+encoder/block3_conv3/BiasAdd/ReadVariableOp¢-encoder/block3_conv3/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv3/Conv2D/ReadVariableOp¢,encoder/block3_conv3/Conv2D_1/ReadVariableOp¢+encoder/block3_conv4/BiasAdd/ReadVariableOp¢-encoder/block3_conv4/BiasAdd_1/ReadVariableOp¢*encoder/block3_conv4/Conv2D/ReadVariableOp¢,encoder/block3_conv4/Conv2D_1/ReadVariableOp¢+encoder/block4_conv1/BiasAdd/ReadVariableOp¢-encoder/block4_conv1/BiasAdd_1/ReadVariableOp¢*encoder/block4_conv1/Conv2D/ReadVariableOp¢,encoder/block4_conv1/Conv2D_1/ReadVariableOpn
tf.reverse_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_4/ReverseV2	ReverseV2inputs_1$tf.reverse_4/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
tf.reverse_3/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_3/ReverseV2	ReverseV2inputs_0$tf.reverse_3/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_4/strided_sliceStridedSlicetf.reverse_4/ReverseV2:output:07tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_3/strided_sliceStridedSlicetf.reverse_3/ReverseV2:output:07tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask±
tf.nn.bias_add_4/BiasAddBiasAdd1tf.__operators__.getitem_4/strided_slice:output:0tf_nn_bias_add_4_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
*encoder/block1_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0à
encoder/block1_conv1/Conv2DConv2D!tf.nn.bias_add_3/BiasAdd:output:02encoder/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+encoder/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¾
encoder/block1_conv1/BiasAddBiasAdd$encoder/block1_conv1/Conv2D:output:03encoder/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv1/ReluRelu%encoder/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
*encoder/block1_conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0æ
encoder/block1_conv2/Conv2DConv2D'encoder/block1_conv1/Relu:activations:02encoder/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

+encoder/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¾
encoder/block1_conv2/BiasAddBiasAdd$encoder/block1_conv2/Conv2D:output:03encoder/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv2/ReluRelu%encoder/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
encoder/block1_pool/MaxPoolMaxPool'encoder/block1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
§
*encoder/block2_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ä
encoder/block2_conv1/Conv2DConv2D$encoder/block1_pool/MaxPool:output:02encoder/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+encoder/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¿
encoder/block2_conv1/BiasAddBiasAdd$encoder/block2_conv1/Conv2D:output:03encoder/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv1/ReluRelu%encoder/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ¨
*encoder/block2_conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ç
encoder/block2_conv2/Conv2DConv2D'encoder/block2_conv1/Relu:activations:02encoder/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

+encoder/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¿
encoder/block2_conv2/BiasAddBiasAdd$encoder/block2_conv2/Conv2D:output:03encoder/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv2/ReluRelu%encoder/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ½
encoder/block2_pool/MaxPoolMaxPool'encoder/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides
¨
*encoder/block3_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0â
encoder/block3_conv1/Conv2DConv2D$encoder/block2_pool/MaxPool:output:02encoder/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv1/BiasAddBiasAdd$encoder/block3_conv1/Conv2D:output:03encoder/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv1/ReluRelu%encoder/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¨
*encoder/block3_conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
encoder/block3_conv2/Conv2DConv2D'encoder/block3_conv1/Relu:activations:02encoder/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv2/BiasAddBiasAdd$encoder/block3_conv2/Conv2D:output:03encoder/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv2/ReluRelu%encoder/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¨
*encoder/block3_conv3/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
encoder/block3_conv3/Conv2DConv2D'encoder/block3_conv2/Relu:activations:02encoder/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv3/BiasAddBiasAdd$encoder/block3_conv3/Conv2D:output:03encoder/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv3/ReluRelu%encoder/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¨
*encoder/block3_conv4/Conv2D/ReadVariableOpReadVariableOp3encoder_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
encoder/block3_conv4/Conv2DConv2D'encoder/block3_conv3/Relu:activations:02encoder/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

+encoder/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp4encoder_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block3_conv4/BiasAddBiasAdd$encoder/block3_conv4/Conv2D:output:03encoder/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv4/ReluRelu%encoder/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@½
encoder/block3_pool/MaxPoolMaxPool'encoder/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
¨
*encoder/block4_conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0â
encoder/block4_conv1/Conv2DConv2D$encoder/block3_pool/MaxPool:output:02encoder/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

+encoder/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
encoder/block4_conv1/BiasAddBiasAdd$encoder/block4_conv1/Conv2D:output:03encoder/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
encoder/block4_conv1/ReluRelu%encoder/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¨
,encoder/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ä
encoder/block1_conv1/Conv2D_1Conv2D!tf.nn.bias_add_4/BiasAdd:output:04encoder/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

-encoder/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ä
encoder/block1_conv1/BiasAdd_1BiasAdd&encoder/block1_conv1/Conv2D_1:output:05encoder/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv1/Relu_1Relu'encoder/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
,encoder/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ì
encoder/block1_conv2/Conv2D_1Conv2D)encoder/block1_conv1/Relu_1:activations:04encoder/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

-encoder/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ä
encoder/block1_conv2/BiasAdd_1BiasAdd&encoder/block1_conv2/Conv2D_1:output:05encoder/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
encoder/block1_conv2/Relu_1Relu'encoder/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
encoder/block1_pool/MaxPool_1MaxPool)encoder/block1_conv2/Relu_1:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
©
,encoder/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ê
encoder/block2_conv1/Conv2D_1Conv2D&encoder/block1_pool/MaxPool_1:output:04encoder/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

-encoder/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Å
encoder/block2_conv1/BiasAdd_1BiasAdd&encoder/block2_conv1/Conv2D_1:output:05encoder/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv1/Relu_1Relu'encoder/block2_conv1/BiasAdd_1:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿª
,encoder/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0í
encoder/block2_conv2/Conv2D_1Conv2D)encoder/block2_conv1/Relu_1:activations:04encoder/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

-encoder/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Å
encoder/block2_conv2/BiasAdd_1BiasAdd&encoder/block2_conv2/Conv2D_1:output:05encoder/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
encoder/block2_conv2/Relu_1Relu'encoder/block2_conv2/BiasAdd_1:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÁ
encoder/block2_pool/MaxPool_1MaxPool)encoder/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides
ª
,encoder/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0è
encoder/block3_conv1/Conv2D_1Conv2D&encoder/block2_pool/MaxPool_1:output:04encoder/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv1/BiasAdd_1BiasAdd&encoder/block3_conv1/Conv2D_1:output:05encoder/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv1/Relu_1Relu'encoder/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ª
,encoder/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
encoder/block3_conv2/Conv2D_1Conv2D)encoder/block3_conv1/Relu_1:activations:04encoder/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv2/BiasAdd_1BiasAdd&encoder/block3_conv2/Conv2D_1:output:05encoder/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv2/Relu_1Relu'encoder/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ª
,encoder/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
encoder/block3_conv3/Conv2D_1Conv2D)encoder/block3_conv2/Relu_1:activations:04encoder/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv3/BiasAdd_1BiasAdd&encoder/block3_conv3/Conv2D_1:output:05encoder/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv3/Relu_1Relu'encoder/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ª
,encoder/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
encoder/block3_conv4/Conv2D_1Conv2D)encoder/block3_conv3/Relu_1:activations:04encoder/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides

-encoder/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block3_conv4/BiasAdd_1BiasAdd&encoder/block3_conv4/Conv2D_1:output:05encoder/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
encoder/block3_conv4/Relu_1Relu'encoder/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@Á
encoder/block3_pool/MaxPool_1MaxPool)encoder/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
ª
,encoder/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOp3encoder_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0è
encoder/block4_conv1/Conv2D_1Conv2D&encoder/block3_pool/MaxPool_1:output:04encoder/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

-encoder/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOp4encoder_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
encoder/block4_conv1/BiasAdd_1BiasAdd&encoder/block4_conv1/Conv2D_1:output:05encoder/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
encoder/block4_conv1/Relu_1Relu'encoder/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  x
'ada_in_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ä
ada_in_1/moments/meanMean'encoder/block4_conv1/Relu:activations:00ada_in_1/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
ada_in_1/moments/StopGradientStopGradientada_in_1/moments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
"ada_in_1/moments/SquaredDifferenceSquaredDifference'encoder/block4_conv1/Relu:activations:0&ada_in_1/moments/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  |
+ada_in_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ë
ada_in_1/moments/varianceMean&ada_in_1/moments/SquaredDifference:z:04ada_in_1/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(z
)ada_in_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ê
ada_in_1/moments_1/meanMean)encoder/block4_conv1/Relu_1:activations:02ada_in_1/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
ada_in_1/moments_1/StopGradientStopGradient ada_in_1/moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
$ada_in_1/moments_1/SquaredDifferenceSquaredDifference)encoder/block4_conv1/Relu_1:activations:0(ada_in_1/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ~
-ada_in_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ñ
ada_in_1/moments_1/varianceMean(ada_in_1/moments_1/SquaredDifference:z:06ada_in_1/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(S
ada_in_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
ada_in_1/addAddV2"ada_in_1/moments/variance:output:0ada_in_1/add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
ada_in_1/SqrtSqrtada_in_1/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
ada_in_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
ada_in_1/add_1AddV2$ada_in_1/moments_1/variance:output:0ada_in_1/add_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
ada_in_1/Sqrt_1Sqrtada_in_1/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ada_in_1/subSub'encoder/block4_conv1/Relu:activations:0ada_in_1/moments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  u
ada_in_1/mulMulada_in_1/Sqrt_1:y:0ada_in_1/sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  {
ada_in_1/truedivRealDivada_in_1/mul:z:0ada_in_1/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
ada_in_1/add_2AddV2ada_in_1/truediv:z:0 ada_in_1/moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ï
(decoder/conv2d_9/StatefulPartitionedCallStatefulPartitionedCallada_in_1/add_2:z:0decoder_conv2d_9_11134decoder_conv2d_9_11136*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7882n
decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"        p
decoder/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d_3/mulMul&decoder/up_sampling2d_3/Const:output:0(decoder/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:ö
4decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor1decoder/conv2d_9/StatefulPartitionedCall:output:0decoder/up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(¥
)decoder/conv2d_10/StatefulPartitionedCallStatefulPartitionedCallEdecoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0decoder_conv2d_10_11143decoder_conv2d_10_11145*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7904
)decoder/conv2d_11/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_10/StatefulPartitionedCall:output:0decoder_conv2d_11_11148decoder_conv2d_11_11150*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7922
)decoder/conv2d_12/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_11/StatefulPartitionedCall:output:0decoder_conv2d_12_11153decoder_conv2d_12_11155*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7940
)decoder/conv2d_13/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_12/StatefulPartitionedCall:output:0decoder_conv2d_13_11158decoder_conv2d_13_11160*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7958n
decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   p
decoder/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d_4/mulMul&decoder/up_sampling2d_4/Const:output:0(decoder/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:ù
4decoder/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor2decoder/conv2d_13/StatefulPartitionedCall:output:0decoder/up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(§
)decoder/conv2d_14/StatefulPartitionedCallStatefulPartitionedCallEdecoder/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0decoder_conv2d_14_11167decoder_conv2d_14_11169*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7980
)decoder/conv2d_15/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_14/StatefulPartitionedCall:output:0decoder_conv2d_15_11172decoder_conv2d_15_11174*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7998n
decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d_5/mulMul&decoder/up_sampling2d_5/Const:output:0(decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:ø
4decoder/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor2decoder/conv2d_15/StatefulPartitionedCall:output:0decoder/up_sampling2d_5/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(¦
)decoder/conv2d_16/StatefulPartitionedCallStatefulPartitionedCallEdecoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0decoder_conv2d_16_11181decoder_conv2d_16_11183*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8020
)decoder/conv2d_17/StatefulPartitionedCallStatefulPartitionedCall2decoder/conv2d_16/StatefulPartitionedCall:output:0decoder_conv2d_17_11186decoder_conv2d_17_11188*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8037o
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CØ
(tf.clip_by_value_1/clip_by_value/MinimumMinimum2decoder/conv2d_17/StatefulPartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Â
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity$tf.clip_by_value_1/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
NoOpNoOp*^decoder/conv2d_10/StatefulPartitionedCall*^decoder/conv2d_11/StatefulPartitionedCall*^decoder/conv2d_12/StatefulPartitionedCall*^decoder/conv2d_13/StatefulPartitionedCall*^decoder/conv2d_14/StatefulPartitionedCall*^decoder/conv2d_15/StatefulPartitionedCall*^decoder/conv2d_16/StatefulPartitionedCall*^decoder/conv2d_17/StatefulPartitionedCall)^decoder/conv2d_9/StatefulPartitionedCall,^encoder/block1_conv1/BiasAdd/ReadVariableOp.^encoder/block1_conv1/BiasAdd_1/ReadVariableOp+^encoder/block1_conv1/Conv2D/ReadVariableOp-^encoder/block1_conv1/Conv2D_1/ReadVariableOp,^encoder/block1_conv2/BiasAdd/ReadVariableOp.^encoder/block1_conv2/BiasAdd_1/ReadVariableOp+^encoder/block1_conv2/Conv2D/ReadVariableOp-^encoder/block1_conv2/Conv2D_1/ReadVariableOp,^encoder/block2_conv1/BiasAdd/ReadVariableOp.^encoder/block2_conv1/BiasAdd_1/ReadVariableOp+^encoder/block2_conv1/Conv2D/ReadVariableOp-^encoder/block2_conv1/Conv2D_1/ReadVariableOp,^encoder/block2_conv2/BiasAdd/ReadVariableOp.^encoder/block2_conv2/BiasAdd_1/ReadVariableOp+^encoder/block2_conv2/Conv2D/ReadVariableOp-^encoder/block2_conv2/Conv2D_1/ReadVariableOp,^encoder/block3_conv1/BiasAdd/ReadVariableOp.^encoder/block3_conv1/BiasAdd_1/ReadVariableOp+^encoder/block3_conv1/Conv2D/ReadVariableOp-^encoder/block3_conv1/Conv2D_1/ReadVariableOp,^encoder/block3_conv2/BiasAdd/ReadVariableOp.^encoder/block3_conv2/BiasAdd_1/ReadVariableOp+^encoder/block3_conv2/Conv2D/ReadVariableOp-^encoder/block3_conv2/Conv2D_1/ReadVariableOp,^encoder/block3_conv3/BiasAdd/ReadVariableOp.^encoder/block3_conv3/BiasAdd_1/ReadVariableOp+^encoder/block3_conv3/Conv2D/ReadVariableOp-^encoder/block3_conv3/Conv2D_1/ReadVariableOp,^encoder/block3_conv4/BiasAdd/ReadVariableOp.^encoder/block3_conv4/BiasAdd_1/ReadVariableOp+^encoder/block3_conv4/Conv2D/ReadVariableOp-^encoder/block3_conv4/Conv2D_1/ReadVariableOp,^encoder/block4_conv1/BiasAdd/ReadVariableOp.^encoder/block4_conv1/BiasAdd_1/ReadVariableOp+^encoder/block4_conv1/Conv2D/ReadVariableOp-^encoder/block4_conv1/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)decoder/conv2d_10/StatefulPartitionedCall)decoder/conv2d_10/StatefulPartitionedCall2V
)decoder/conv2d_11/StatefulPartitionedCall)decoder/conv2d_11/StatefulPartitionedCall2V
)decoder/conv2d_12/StatefulPartitionedCall)decoder/conv2d_12/StatefulPartitionedCall2V
)decoder/conv2d_13/StatefulPartitionedCall)decoder/conv2d_13/StatefulPartitionedCall2V
)decoder/conv2d_14/StatefulPartitionedCall)decoder/conv2d_14/StatefulPartitionedCall2V
)decoder/conv2d_15/StatefulPartitionedCall)decoder/conv2d_15/StatefulPartitionedCall2V
)decoder/conv2d_16/StatefulPartitionedCall)decoder/conv2d_16/StatefulPartitionedCall2V
)decoder/conv2d_17/StatefulPartitionedCall)decoder/conv2d_17/StatefulPartitionedCall2T
(decoder/conv2d_9/StatefulPartitionedCall(decoder/conv2d_9/StatefulPartitionedCall2Z
+encoder/block1_conv1/BiasAdd/ReadVariableOp+encoder/block1_conv1/BiasAdd/ReadVariableOp2^
-encoder/block1_conv1/BiasAdd_1/ReadVariableOp-encoder/block1_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block1_conv1/Conv2D/ReadVariableOp*encoder/block1_conv1/Conv2D/ReadVariableOp2\
,encoder/block1_conv1/Conv2D_1/ReadVariableOp,encoder/block1_conv1/Conv2D_1/ReadVariableOp2Z
+encoder/block1_conv2/BiasAdd/ReadVariableOp+encoder/block1_conv2/BiasAdd/ReadVariableOp2^
-encoder/block1_conv2/BiasAdd_1/ReadVariableOp-encoder/block1_conv2/BiasAdd_1/ReadVariableOp2X
*encoder/block1_conv2/Conv2D/ReadVariableOp*encoder/block1_conv2/Conv2D/ReadVariableOp2\
,encoder/block1_conv2/Conv2D_1/ReadVariableOp,encoder/block1_conv2/Conv2D_1/ReadVariableOp2Z
+encoder/block2_conv1/BiasAdd/ReadVariableOp+encoder/block2_conv1/BiasAdd/ReadVariableOp2^
-encoder/block2_conv1/BiasAdd_1/ReadVariableOp-encoder/block2_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block2_conv1/Conv2D/ReadVariableOp*encoder/block2_conv1/Conv2D/ReadVariableOp2\
,encoder/block2_conv1/Conv2D_1/ReadVariableOp,encoder/block2_conv1/Conv2D_1/ReadVariableOp2Z
+encoder/block2_conv2/BiasAdd/ReadVariableOp+encoder/block2_conv2/BiasAdd/ReadVariableOp2^
-encoder/block2_conv2/BiasAdd_1/ReadVariableOp-encoder/block2_conv2/BiasAdd_1/ReadVariableOp2X
*encoder/block2_conv2/Conv2D/ReadVariableOp*encoder/block2_conv2/Conv2D/ReadVariableOp2\
,encoder/block2_conv2/Conv2D_1/ReadVariableOp,encoder/block2_conv2/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv1/BiasAdd/ReadVariableOp+encoder/block3_conv1/BiasAdd/ReadVariableOp2^
-encoder/block3_conv1/BiasAdd_1/ReadVariableOp-encoder/block3_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv1/Conv2D/ReadVariableOp*encoder/block3_conv1/Conv2D/ReadVariableOp2\
,encoder/block3_conv1/Conv2D_1/ReadVariableOp,encoder/block3_conv1/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv2/BiasAdd/ReadVariableOp+encoder/block3_conv2/BiasAdd/ReadVariableOp2^
-encoder/block3_conv2/BiasAdd_1/ReadVariableOp-encoder/block3_conv2/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv2/Conv2D/ReadVariableOp*encoder/block3_conv2/Conv2D/ReadVariableOp2\
,encoder/block3_conv2/Conv2D_1/ReadVariableOp,encoder/block3_conv2/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv3/BiasAdd/ReadVariableOp+encoder/block3_conv3/BiasAdd/ReadVariableOp2^
-encoder/block3_conv3/BiasAdd_1/ReadVariableOp-encoder/block3_conv3/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv3/Conv2D/ReadVariableOp*encoder/block3_conv3/Conv2D/ReadVariableOp2\
,encoder/block3_conv3/Conv2D_1/ReadVariableOp,encoder/block3_conv3/Conv2D_1/ReadVariableOp2Z
+encoder/block3_conv4/BiasAdd/ReadVariableOp+encoder/block3_conv4/BiasAdd/ReadVariableOp2^
-encoder/block3_conv4/BiasAdd_1/ReadVariableOp-encoder/block3_conv4/BiasAdd_1/ReadVariableOp2X
*encoder/block3_conv4/Conv2D/ReadVariableOp*encoder/block3_conv4/Conv2D/ReadVariableOp2\
,encoder/block3_conv4/Conv2D_1/ReadVariableOp,encoder/block3_conv4/Conv2D_1/ReadVariableOp2Z
+encoder/block4_conv1/BiasAdd/ReadVariableOp+encoder/block4_conv1/BiasAdd/ReadVariableOp2^
-encoder/block4_conv1/BiasAdd_1/ReadVariableOp-encoder/block4_conv1/BiasAdd_1/ReadVariableOp2X
*encoder/block4_conv1/Conv2D/ReadVariableOp*encoder/block4_conv1/Conv2D/ReadVariableOp2\
,encoder/block4_conv1/Conv2D_1/ReadVariableOp,encoder/block4_conv1/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1: 

_output_shapes
:: 

_output_shapes
:
Ñ
Ê
__inference_call_12323

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ö
Ç
__inference_call_7998

inputs9
conv2d_readvariableop_resource:@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_11973

inputs
identity
MaxPoolMaxPoolinputs*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
r
IdentityIdentityMaxPool:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ö?

A__inference_model_2_layer_call_and_return_conditional_losses_9638

inputs
inputs_1!
tf_nn_bias_add_4_biasadd_bias!
tf_nn_bias_add_3_biasadd_bias&
encoder_9505:@
encoder_9507:@&
encoder_9509:@@
encoder_9511:@'
encoder_9513:@
encoder_9515:	(
encoder_9517:
encoder_9519:	(
encoder_9521:
encoder_9523:	(
encoder_9525:
encoder_9527:	(
encoder_9529:
encoder_9531:	(
encoder_9533:
encoder_9535:	(
encoder_9537:
encoder_9539:	(
decoder_9596:
decoder_9598:	(
decoder_9600:
decoder_9602:	(
decoder_9604:
decoder_9606:	(
decoder_9608:
decoder_9610:	(
decoder_9612:
decoder_9614:	(
decoder_9616:
decoder_9618:	'
decoder_9620:@
decoder_9622:@&
decoder_9624:@@
decoder_9626:@&
decoder_9628:@
decoder_9630:
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall¢!encoder/StatefulPartitionedCall_1n
tf.reverse_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_4/ReverseV2	ReverseV2inputs_1$tf.reverse_4/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
tf.reverse_3/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_3/ReverseV2	ReverseV2inputs$tf.reverse_3/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_4/strided_sliceStridedSlicetf.reverse_4/ReverseV2:output:07tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_3/strided_sliceStridedSlicetf.reverse_3/ReverseV2:output:07tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask±
tf.nn.bias_add_4/BiasAddBiasAdd1tf.__operators__.getitem_4/strided_slice:output:0tf_nn_bias_add_4_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
encoder/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0encoder_9505encoder_9507encoder_9509encoder_9511encoder_9513encoder_9515encoder_9517encoder_9519encoder_9521encoder_9523encoder_9525encoder_9527encoder_9529encoder_9531encoder_9533encoder_9535encoder_9537encoder_9539*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9504ê
!encoder/StatefulPartitionedCall_1StatefulPartitionedCall!tf.nn.bias_add_4/BiasAdd:output:0encoder_9505encoder_9507encoder_9509encoder_9511encoder_9513encoder_9515encoder_9517encoder_9519encoder_9521encoder_9523encoder_9525encoder_9527encoder_9529encoder_9531encoder_9533encoder_9535encoder_9537encoder_9539*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9504
ada_in_1/PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:3*encoder/StatefulPartitionedCall_1:output:3*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_ada_in_1_layer_call_and_return_conditional_losses_9594
decoder/StatefulPartitionedCallStatefulPartitionedCall!ada_in_1/PartitionedCall:output:0decoder_9596decoder_9598decoder_9600decoder_9602decoder_9604decoder_9606decoder_9608decoder_9610decoder_9612decoder_9614decoder_9616decoder_9618decoder_9620decoder_9622decoder_9624decoder_9626decoder_9628decoder_9630*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_8984o
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÎ
(tf.clip_by_value_1/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Â
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity$tf.clip_by_value_1/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall"^encoder/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2F
!encoder/StatefulPartitionedCall_1!encoder/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
»
¡
,__inference_block1_conv1_layer_call_fn_11922

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_8102
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_block1_pool_layer_call_and_return_conditional_losses_8057

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
È
__inference_call_12458

inputs9
conv2d_readvariableop_resource:@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

õ
D__inference_conv2d_16_layer_call_and_return_conditional_losses_12510

inputs8
conv2d_readvariableop_resource:@@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
¤
,__inference_block3_conv4_layer_call_fn_12102

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_8233
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
T
(__inference_ada_in_1_layer_call_fn_11682
inputs_0
inputs_1
identityÆ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_ada_in_1_layer_call_and_return_conditional_losses_9594i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1
ñ@
º
B__inference_model_2_layer_call_and_return_conditional_losses_10512
content_image
style_image!
tf_nn_bias_add_4_biasadd_bias!
tf_nn_bias_add_3_biasadd_bias'
encoder_10407:@
encoder_10409:@'
encoder_10411:@@
encoder_10413:@(
encoder_10415:@
encoder_10417:	)
encoder_10419:
encoder_10421:	)
encoder_10423:
encoder_10425:	)
encoder_10427:
encoder_10429:	)
encoder_10431:
encoder_10433:	)
encoder_10435:
encoder_10437:	)
encoder_10439:
encoder_10441:	)
decoder_10470:
decoder_10472:	)
decoder_10474:
decoder_10476:	)
decoder_10478:
decoder_10480:	)
decoder_10482:
decoder_10484:	)
decoder_10486:
decoder_10488:	)
decoder_10490:
decoder_10492:	(
decoder_10494:@
decoder_10496:@'
decoder_10498:@@
decoder_10500:@'
decoder_10502:@
decoder_10504:
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall¢!encoder/StatefulPartitionedCall_1n
tf.reverse_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_4/ReverseV2	ReverseV2style_image$tf.reverse_4/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
tf.reverse_3/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
tf.reverse_3/ReverseV2	ReverseV2content_image$tf.reverse_3/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_4/strided_sliceStridedSlicetf.reverse_4/ReverseV2:output:07tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
(tf.__operators__.getitem_3/strided_sliceStridedSlicetf.reverse_3/ReverseV2:output:07tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask±
tf.nn.bias_add_4/BiasAddBiasAdd1tf.__operators__.getitem_4/strided_slice:output:0tf_nn_bias_add_4_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
encoder/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0encoder_10407encoder_10409encoder_10411encoder_10413encoder_10415encoder_10417encoder_10419encoder_10421encoder_10423encoder_10425encoder_10427encoder_10429encoder_10431encoder_10433encoder_10435encoder_10437encoder_10439encoder_10441*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9846ü
!encoder/StatefulPartitionedCall_1StatefulPartitionedCall!tf.nn.bias_add_4/BiasAdd:output:0encoder_10407encoder_10409encoder_10411encoder_10413encoder_10415encoder_10417encoder_10419encoder_10421encoder_10423encoder_10425encoder_10427encoder_10429encoder_10431encoder_10433encoder_10435encoder_10437encoder_10439encoder_10441*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapesu
s:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@:ÿÿÿÿÿÿÿÿÿ  *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_9846
ada_in_1/PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:3*encoder/StatefulPartitionedCall_1:output:3*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_ada_in_1_layer_call_and_return_conditional_losses_9594 
decoder/StatefulPartitionedCallStatefulPartitionedCall!ada_in_1/PartitionedCall:output:0decoder_10470decoder_10472decoder_10474decoder_10476decoder_10478decoder_10480decoder_10482decoder_10484decoder_10486decoder_10488decoder_10490decoder_10492decoder_10494decoder_10496decoder_10498decoder_10500decoder_10502decoder_10504*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_9223o
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÎ
(tf.clip_by_value_1/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Â
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity$tf.clip_by_value_1/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall"^encoder/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2F
!encoder/StatefulPartitionedCall_1!encoder/StatefulPartitionedCall_1:` \
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecontent_image:^Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namestyle_image: 

_output_shapes
:: 

_output_shapes
:
û

G__inference_block2_conv1_layer_call_and_return_conditional_losses_11993

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
ô
C__inference_conv2d_16_layer_call_and_return_conditional_losses_8959

inputs8
conv2d_readvariableop_resource:@@)
add_readvariableop_resource:@
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î9
Ä
A__inference_decoder_layer_call_and_return_conditional_losses_9407
conv2d_9_input)
conv2d_9_9358:
conv2d_9_9360:	*
conv2d_10_9364:
conv2d_10_9366:	*
conv2d_11_9369:
conv2d_11_9371:	*
conv2d_12_9374:
conv2d_12_9376:	*
conv2d_13_9379:
conv2d_13_9381:	*
conv2d_14_9385:
conv2d_14_9387:	)
conv2d_15_9390:@
conv2d_15_9392:@(
conv2d_16_9396:@@
conv2d_16_9398:@(
conv2d_17_9401:@
conv2d_17_9403:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallþ
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputconv2d_9_9358conv2d_9_9360*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_8799ó
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8812
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_10_9364conv2d_10_9366*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_8827
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_9369conv2d_11_9371*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_8846
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_9374conv2d_12_9376*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_8865
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_9379conv2d_13_9381*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_8884ö
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8897
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_9385conv2d_14_9387*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_8912
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_9390conv2d_15_9392*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_8931õ
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8944
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_16_9396conv2d_16_9398*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_8959
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_9401conv2d_17_9403*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_8977
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
(
_user_specified_nameconv2d_9_input
ÿ
ø
D__inference_conv2d_10_layer_call_and_return_conditional_losses_12240

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8757

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Ê
__inference_call_12253

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

a
E__inference_block3_pool_layer_call_and_return_conditional_losses_8081

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

G__inference_block3_conv1_layer_call_and_return_conditional_losses_12053

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
ö
B__inference_conv2d_9_layer_call_and_return_conditional_losses_8799

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

­
&__inference_decoder_layer_call_fn_9303
conv2d_9_input#
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	%

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@$

unknown_15:@

unknown_16:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_9223y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
(
_user_specified_nameconv2d_9_input
î
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_12033

inputs
identity
MaxPoolMaxPoolinputs*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
s
IdentityIdentityMaxPool:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸á
±'
__inference__wrapped_model_8048
content_image
style_image)
%model_2_tf_nn_bias_add_4_biasadd_bias)
%model_2_tf_nn_bias_add_3_biasadd_biasU
;model_2_encoder_block1_conv1_conv2d_readvariableop_resource:@J
<model_2_encoder_block1_conv1_biasadd_readvariableop_resource:@U
;model_2_encoder_block1_conv2_conv2d_readvariableop_resource:@@J
<model_2_encoder_block1_conv2_biasadd_readvariableop_resource:@V
;model_2_encoder_block2_conv1_conv2d_readvariableop_resource:@K
<model_2_encoder_block2_conv1_biasadd_readvariableop_resource:	W
;model_2_encoder_block2_conv2_conv2d_readvariableop_resource:K
<model_2_encoder_block2_conv2_biasadd_readvariableop_resource:	W
;model_2_encoder_block3_conv1_conv2d_readvariableop_resource:K
<model_2_encoder_block3_conv1_biasadd_readvariableop_resource:	W
;model_2_encoder_block3_conv2_conv2d_readvariableop_resource:K
<model_2_encoder_block3_conv2_biasadd_readvariableop_resource:	W
;model_2_encoder_block3_conv3_conv2d_readvariableop_resource:K
<model_2_encoder_block3_conv3_biasadd_readvariableop_resource:	W
;model_2_encoder_block3_conv4_conv2d_readvariableop_resource:K
<model_2_encoder_block3_conv4_biasadd_readvariableop_resource:	W
;model_2_encoder_block4_conv1_conv2d_readvariableop_resource:K
<model_2_encoder_block4_conv1_biasadd_readvariableop_resource:	9
model_2_decoder_conv2d_9_7883:,
model_2_decoder_conv2d_9_7885:	:
model_2_decoder_conv2d_10_7905:-
model_2_decoder_conv2d_10_7907:	:
model_2_decoder_conv2d_11_7923:-
model_2_decoder_conv2d_11_7925:	:
model_2_decoder_conv2d_12_7941:-
model_2_decoder_conv2d_12_7943:	:
model_2_decoder_conv2d_13_7959:-
model_2_decoder_conv2d_13_7961:	:
model_2_decoder_conv2d_14_7981:-
model_2_decoder_conv2d_14_7983:	9
model_2_decoder_conv2d_15_7999:@,
model_2_decoder_conv2d_15_8001:@8
model_2_decoder_conv2d_16_8021:@@,
model_2_decoder_conv2d_16_8023:@8
model_2_decoder_conv2d_17_8038:@,
model_2_decoder_conv2d_17_8040:
identity¢1model_2/decoder/conv2d_10/StatefulPartitionedCall¢1model_2/decoder/conv2d_11/StatefulPartitionedCall¢1model_2/decoder/conv2d_12/StatefulPartitionedCall¢1model_2/decoder/conv2d_13/StatefulPartitionedCall¢1model_2/decoder/conv2d_14/StatefulPartitionedCall¢1model_2/decoder/conv2d_15/StatefulPartitionedCall¢1model_2/decoder/conv2d_16/StatefulPartitionedCall¢1model_2/decoder/conv2d_17/StatefulPartitionedCall¢0model_2/decoder/conv2d_9/StatefulPartitionedCall¢3model_2/encoder/block1_conv1/BiasAdd/ReadVariableOp¢5model_2/encoder/block1_conv1/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block1_conv1/Conv2D/ReadVariableOp¢4model_2/encoder/block1_conv1/Conv2D_1/ReadVariableOp¢3model_2/encoder/block1_conv2/BiasAdd/ReadVariableOp¢5model_2/encoder/block1_conv2/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block1_conv2/Conv2D/ReadVariableOp¢4model_2/encoder/block1_conv2/Conv2D_1/ReadVariableOp¢3model_2/encoder/block2_conv1/BiasAdd/ReadVariableOp¢5model_2/encoder/block2_conv1/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block2_conv1/Conv2D/ReadVariableOp¢4model_2/encoder/block2_conv1/Conv2D_1/ReadVariableOp¢3model_2/encoder/block2_conv2/BiasAdd/ReadVariableOp¢5model_2/encoder/block2_conv2/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block2_conv2/Conv2D/ReadVariableOp¢4model_2/encoder/block2_conv2/Conv2D_1/ReadVariableOp¢3model_2/encoder/block3_conv1/BiasAdd/ReadVariableOp¢5model_2/encoder/block3_conv1/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block3_conv1/Conv2D/ReadVariableOp¢4model_2/encoder/block3_conv1/Conv2D_1/ReadVariableOp¢3model_2/encoder/block3_conv2/BiasAdd/ReadVariableOp¢5model_2/encoder/block3_conv2/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block3_conv2/Conv2D/ReadVariableOp¢4model_2/encoder/block3_conv2/Conv2D_1/ReadVariableOp¢3model_2/encoder/block3_conv3/BiasAdd/ReadVariableOp¢5model_2/encoder/block3_conv3/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block3_conv3/Conv2D/ReadVariableOp¢4model_2/encoder/block3_conv3/Conv2D_1/ReadVariableOp¢3model_2/encoder/block3_conv4/BiasAdd/ReadVariableOp¢5model_2/encoder/block3_conv4/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block3_conv4/Conv2D/ReadVariableOp¢4model_2/encoder/block3_conv4/Conv2D_1/ReadVariableOp¢3model_2/encoder/block4_conv1/BiasAdd/ReadVariableOp¢5model_2/encoder/block4_conv1/BiasAdd_1/ReadVariableOp¢2model_2/encoder/block4_conv1/Conv2D/ReadVariableOp¢4model_2/encoder/block4_conv1/Conv2D_1/ReadVariableOpv
#model_2/tf.reverse_4/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¢
model_2/tf.reverse_4/ReverseV2	ReverseV2style_image,model_2/tf.reverse_4/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
#model_2/tf.reverse_3/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¤
model_2/tf.reverse_3/ReverseV2	ReverseV2content_image,model_2/tf.reverse_3/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6model_2/tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
8model_2/tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
8model_2/tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ°
0model_2/tf.__operators__.getitem_4/strided_sliceStridedSlice'model_2/tf.reverse_4/ReverseV2:output:0?model_2/tf.__operators__.getitem_4/strided_slice/stack:output:0Amodel_2/tf.__operators__.getitem_4/strided_slice/stack_1:output:0Amodel_2/tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_mask
6model_2/tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
8model_2/tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
8model_2/tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ°
0model_2/tf.__operators__.getitem_3/strided_sliceStridedSlice'model_2/tf.reverse_3/ReverseV2:output:0?model_2/tf.__operators__.getitem_3/strided_slice/stack:output:0Amodel_2/tf.__operators__.getitem_3/strided_slice/stack_1:output:0Amodel_2/tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
ellipsis_mask*
end_maskÉ
 model_2/tf.nn.bias_add_4/BiasAddBiasAdd9model_2/tf.__operators__.getitem_4/strided_slice:output:0%model_2_tf_nn_bias_add_4_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
 model_2/tf.nn.bias_add_3/BiasAddBiasAdd9model_2/tf.__operators__.getitem_3/strided_slice:output:0%model_2_tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_2/encoder/block1_conv1/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ø
#model_2/encoder/block1_conv1/Conv2DConv2D)model_2/tf.nn.bias_add_3/BiasAdd:output:0:model_2/encoder/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
¬
3model_2/encoder/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
$model_2/encoder/block1_conv1/BiasAddBiasAdd,model_2/encoder/block1_conv1/Conv2D:output:0;model_2/encoder/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!model_2/encoder/block1_conv1/ReluRelu-model_2/encoder/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
2model_2/encoder/block1_conv2/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0þ
#model_2/encoder/block1_conv2/Conv2DConv2D/model_2/encoder/block1_conv1/Relu:activations:0:model_2/encoder/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
¬
3model_2/encoder/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
$model_2/encoder/block1_conv2/BiasAddBiasAdd,model_2/encoder/block1_conv2/Conv2D:output:0;model_2/encoder/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!model_2/encoder/block1_conv2/ReluRelu-model_2/encoder/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
#model_2/encoder/block1_pool/MaxPoolMaxPool/model_2/encoder/block1_conv2/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
·
2model_2/encoder/block2_conv1/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ü
#model_2/encoder/block2_conv1/Conv2DConv2D,model_2/encoder/block1_pool/MaxPool:output:0:model_2/encoder/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
3model_2/encoder/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0×
$model_2/encoder/block2_conv1/BiasAddBiasAdd,model_2/encoder/block2_conv1/Conv2D:output:0;model_2/encoder/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
!model_2/encoder/block2_conv1/ReluRelu-model_2/encoder/block2_conv1/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ¸
2model_2/encoder/block2_conv2/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ÿ
#model_2/encoder/block2_conv2/Conv2DConv2D/model_2/encoder/block2_conv1/Relu:activations:0:model_2/encoder/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
3model_2/encoder/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0×
$model_2/encoder/block2_conv2/BiasAddBiasAdd,model_2/encoder/block2_conv2/Conv2D:output:0;model_2/encoder/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
!model_2/encoder/block2_conv2/ReluRelu-model_2/encoder/block2_conv2/BiasAdd:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÍ
#model_2/encoder/block2_pool/MaxPoolMaxPool/model_2/encoder/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides
¸
2model_2/encoder/block3_conv1/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_2/encoder/block3_conv1/Conv2DConv2D,model_2/encoder/block2_pool/MaxPool:output:0:model_2/encoder/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
­
3model_2/encoder/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_2/encoder/block3_conv1/BiasAddBiasAdd,model_2/encoder/block3_conv1/Conv2D:output:0;model_2/encoder/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
!model_2/encoder/block3_conv1/ReluRelu-model_2/encoder/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¸
2model_2/encoder/block3_conv2/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_2/encoder/block3_conv2/Conv2DConv2D/model_2/encoder/block3_conv1/Relu:activations:0:model_2/encoder/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
­
3model_2/encoder/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_2/encoder/block3_conv2/BiasAddBiasAdd,model_2/encoder/block3_conv2/Conv2D:output:0;model_2/encoder/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
!model_2/encoder/block3_conv2/ReluRelu-model_2/encoder/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¸
2model_2/encoder/block3_conv3/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_2/encoder/block3_conv3/Conv2DConv2D/model_2/encoder/block3_conv2/Relu:activations:0:model_2/encoder/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
­
3model_2/encoder/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_2/encoder/block3_conv3/BiasAddBiasAdd,model_2/encoder/block3_conv3/Conv2D:output:0;model_2/encoder/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
!model_2/encoder/block3_conv3/ReluRelu-model_2/encoder/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@¸
2model_2/encoder/block3_conv4/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_2/encoder/block3_conv4/Conv2DConv2D/model_2/encoder/block3_conv3/Relu:activations:0:model_2/encoder/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
­
3model_2/encoder/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_2/encoder/block3_conv4/BiasAddBiasAdd,model_2/encoder/block3_conv4/Conv2D:output:0;model_2/encoder/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
!model_2/encoder/block3_conv4/ReluRelu-model_2/encoder/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@Í
#model_2/encoder/block3_pool/MaxPoolMaxPool/model_2/encoder/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
¸
2model_2/encoder/block4_conv1/Conv2D/ReadVariableOpReadVariableOp;model_2_encoder_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_2/encoder/block4_conv1/Conv2DConv2D,model_2/encoder/block3_pool/MaxPool:output:0:model_2/encoder/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
­
3model_2/encoder/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp<model_2_encoder_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_2/encoder/block4_conv1/BiasAddBiasAdd,model_2/encoder/block4_conv1/Conv2D:output:0;model_2/encoder/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!model_2/encoder/block4_conv1/ReluRelu-model_2/encoder/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¸
4model_2/encoder/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ü
%model_2/encoder/block1_conv1/Conv2D_1Conv2D)model_2/tf.nn.bias_add_4/BiasAdd:output:0<model_2/encoder/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
®
5model_2/encoder/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ü
&model_2/encoder/block1_conv1/BiasAdd_1BiasAdd.model_2/encoder/block1_conv1/Conv2D_1:output:0=model_2/encoder/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#model_2/encoder/block1_conv1/Relu_1Relu/model_2/encoder/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¸
4model_2/encoder/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
%model_2/encoder/block1_conv2/Conv2D_1Conv2D1model_2/encoder/block1_conv1/Relu_1:activations:0<model_2/encoder/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
®
5model_2/encoder/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ü
&model_2/encoder/block1_conv2/BiasAdd_1BiasAdd.model_2/encoder/block1_conv2/Conv2D_1:output:0=model_2/encoder/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#model_2/encoder/block1_conv2/Relu_1Relu/model_2/encoder/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ò
%model_2/encoder/block1_pool/MaxPool_1MaxPool1model_2/encoder/block1_conv2/Relu_1:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
¹
4model_2/encoder/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
%model_2/encoder/block2_conv1/Conv2D_1Conv2D.model_2/encoder/block1_pool/MaxPool_1:output:0<model_2/encoder/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
5model_2/encoder/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ý
&model_2/encoder/block2_conv1/BiasAdd_1BiasAdd.model_2/encoder/block2_conv1/Conv2D_1:output:0=model_2/encoder/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
#model_2/encoder/block2_conv1/Relu_1Relu/model_2/encoder/block2_conv1/BiasAdd_1:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿº
4model_2/encoder/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
%model_2/encoder/block2_conv2/Conv2D_1Conv2D1model_2/encoder/block2_conv1/Relu_1:activations:0<model_2/encoder/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
5model_2/encoder/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ý
&model_2/encoder/block2_conv2/BiasAdd_1BiasAdd.model_2/encoder/block2_conv2/Conv2D_1:output:0=model_2/encoder/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
#model_2/encoder/block2_conv2/Relu_1Relu/model_2/encoder/block2_conv2/BiasAdd_1:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿÑ
%model_2/encoder/block2_pool/MaxPool_1MaxPool1model_2/encoder/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
ksize
*
paddingVALID*
strides
º
4model_2/encoder/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
%model_2/encoder/block3_conv1/Conv2D_1Conv2D.model_2/encoder/block2_pool/MaxPool_1:output:0<model_2/encoder/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
¯
5model_2/encoder/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Û
&model_2/encoder/block3_conv1/BiasAdd_1BiasAdd.model_2/encoder/block3_conv1/Conv2D_1:output:0=model_2/encoder/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
#model_2/encoder/block3_conv1/Relu_1Relu/model_2/encoder/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@º
4model_2/encoder/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
%model_2/encoder/block3_conv2/Conv2D_1Conv2D1model_2/encoder/block3_conv1/Relu_1:activations:0<model_2/encoder/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
¯
5model_2/encoder/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Û
&model_2/encoder/block3_conv2/BiasAdd_1BiasAdd.model_2/encoder/block3_conv2/Conv2D_1:output:0=model_2/encoder/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
#model_2/encoder/block3_conv2/Relu_1Relu/model_2/encoder/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@º
4model_2/encoder/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
%model_2/encoder/block3_conv3/Conv2D_1Conv2D1model_2/encoder/block3_conv2/Relu_1:activations:0<model_2/encoder/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
¯
5model_2/encoder/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Û
&model_2/encoder/block3_conv3/BiasAdd_1BiasAdd.model_2/encoder/block3_conv3/Conv2D_1:output:0=model_2/encoder/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
#model_2/encoder/block3_conv3/Relu_1Relu/model_2/encoder/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@º
4model_2/encoder/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
%model_2/encoder/block3_conv4/Conv2D_1Conv2D1model_2/encoder/block3_conv3/Relu_1:activations:0<model_2/encoder/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
¯
5model_2/encoder/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Û
&model_2/encoder/block3_conv4/BiasAdd_1BiasAdd.model_2/encoder/block3_conv4/Conv2D_1:output:0=model_2/encoder/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
#model_2/encoder/block3_conv4/Relu_1Relu/model_2/encoder/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@Ñ
%model_2/encoder/block3_pool/MaxPool_1MaxPool1model_2/encoder/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
º
4model_2/encoder/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOp;model_2_encoder_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
%model_2/encoder/block4_conv1/Conv2D_1Conv2D.model_2/encoder/block3_pool/MaxPool_1:output:0<model_2/encoder/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
¯
5model_2/encoder/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOp<model_2_encoder_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Û
&model_2/encoder/block4_conv1/BiasAdd_1BiasAdd.model_2/encoder/block4_conv1/Conv2D_1:output:0=model_2/encoder/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
#model_2/encoder/block4_conv1/Relu_1Relu/model_2/encoder/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
/model_2/ada_in_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ü
model_2/ada_in_1/moments/meanMean/model_2/encoder/block4_conv1/Relu:activations:08model_2/ada_in_1/moments/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
%model_2/ada_in_1/moments/StopGradientStopGradient&model_2/ada_in_1/moments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
*model_2/ada_in_1/moments/SquaredDifferenceSquaredDifference/model_2/encoder/block4_conv1/Relu:activations:0.model_2/ada_in_1/moments/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
3model_2/ada_in_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ã
!model_2/ada_in_1/moments/varianceMean.model_2/ada_in_1/moments/SquaredDifference:z:0<model_2/ada_in_1/moments/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
1model_2/ada_in_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      â
model_2/ada_in_1/moments_1/meanMean1model_2/encoder/block4_conv1/Relu_1:activations:0:model_2/ada_in_1/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
'model_2/ada_in_1/moments_1/StopGradientStopGradient(model_2/ada_in_1/moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
,model_2/ada_in_1/moments_1/SquaredDifferenceSquaredDifference1model_2/encoder/block4_conv1/Relu_1:activations:00model_2/ada_in_1/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
5model_2/ada_in_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      é
#model_2/ada_in_1/moments_1/varianceMean0model_2/ada_in_1/moments_1/SquaredDifference:z:0>model_2/ada_in_1/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims([
model_2/ada_in_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¥
model_2/ada_in_1/addAddV2*model_2/ada_in_1/moments/variance:output:0model_2/ada_in_1/add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
model_2/ada_in_1/SqrtSqrtmodel_2/ada_in_1/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
model_2/ada_in_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7«
model_2/ada_in_1/add_1AddV2,model_2/ada_in_1/moments_1/variance:output:0!model_2/ada_in_1/add_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
model_2/ada_in_1/Sqrt_1Sqrtmodel_2/ada_in_1/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
model_2/ada_in_1/subSub/model_2/encoder/block4_conv1/Relu:activations:0&model_2/ada_in_1/moments/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
model_2/ada_in_1/mulMulmodel_2/ada_in_1/Sqrt_1:y:0model_2/ada_in_1/sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
model_2/ada_in_1/truedivRealDivmodel_2/ada_in_1/mul:z:0model_2/ada_in_1/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¢
model_2/ada_in_1/add_2AddV2model_2/ada_in_1/truediv:z:0(model_2/ada_in_1/moments_1/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
0model_2/decoder/conv2d_9/StatefulPartitionedCallStatefulPartitionedCallmodel_2/ada_in_1/add_2:z:0model_2_decoder_conv2d_9_7883model_2_decoder_conv2d_9_7885*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7882v
%model_2/decoder/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"        x
'model_2/decoder/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ±
#model_2/decoder/up_sampling2d_3/mulMul.model_2/decoder/up_sampling2d_3/Const:output:00model_2/decoder/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:
<model_2/decoder/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor9model_2/decoder/conv2d_9/StatefulPartitionedCall:output:0'model_2/decoder/up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(Ã
1model_2/decoder/conv2d_10/StatefulPartitionedCallStatefulPartitionedCallMmodel_2/decoder/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0model_2_decoder_conv2d_10_7905model_2_decoder_conv2d_10_7907*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7904°
1model_2/decoder/conv2d_11/StatefulPartitionedCallStatefulPartitionedCall:model_2/decoder/conv2d_10/StatefulPartitionedCall:output:0model_2_decoder_conv2d_11_7923model_2_decoder_conv2d_11_7925*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7922°
1model_2/decoder/conv2d_12/StatefulPartitionedCallStatefulPartitionedCall:model_2/decoder/conv2d_11/StatefulPartitionedCall:output:0model_2_decoder_conv2d_12_7941model_2_decoder_conv2d_12_7943*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7940°
1model_2/decoder/conv2d_13/StatefulPartitionedCallStatefulPartitionedCall:model_2/decoder/conv2d_12/StatefulPartitionedCall:output:0model_2_decoder_conv2d_13_7959model_2_decoder_conv2d_13_7961*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7958v
%model_2/decoder/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   x
'model_2/decoder/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ±
#model_2/decoder/up_sampling2d_4/mulMul.model_2/decoder/up_sampling2d_4/Const:output:00model_2/decoder/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:
<model_2/decoder/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor:model_2/decoder/conv2d_13/StatefulPartitionedCall:output:0'model_2/decoder/up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(Å
1model_2/decoder/conv2d_14/StatefulPartitionedCallStatefulPartitionedCallMmodel_2/decoder/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0model_2_decoder_conv2d_14_7981model_2_decoder_conv2d_14_7983*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7980±
1model_2/decoder/conv2d_15/StatefulPartitionedCallStatefulPartitionedCall:model_2/decoder/conv2d_14/StatefulPartitionedCall:output:0model_2_decoder_conv2d_15_7999model_2_decoder_conv2d_15_8001*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7998v
%model_2/decoder/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      x
'model_2/decoder/up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ±
#model_2/decoder/up_sampling2d_5/mulMul.model_2/decoder/up_sampling2d_5/Const:output:00model_2/decoder/up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:
<model_2/decoder/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor:model_2/decoder/conv2d_15/StatefulPartitionedCall:output:0'model_2/decoder/up_sampling2d_5/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(Ä
1model_2/decoder/conv2d_16/StatefulPartitionedCallStatefulPartitionedCallMmodel_2/decoder/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0model_2_decoder_conv2d_16_8021model_2_decoder_conv2d_16_8023*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8020±
1model_2/decoder/conv2d_17/StatefulPartitionedCallStatefulPartitionedCall:model_2/decoder/conv2d_16/StatefulPartitionedCall:output:0model_2_decoder_conv2d_17_8038model_2_decoder_conv2d_17_8040*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8037w
2model_2/tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cð
0model_2/tf.clip_by_value_1/clip_by_value/MinimumMinimum:model_2/decoder/conv2d_17/StatefulPartitionedCall:output:0;model_2/tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*model_2/tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ú
(model_2/tf.clip_by_value_1/clip_by_valueMaximum4model_2/tf.clip_by_value_1/clip_by_value/Minimum:z:03model_2/tf.clip_by_value_1/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity,model_2/tf.clip_by_value_1/clip_by_value:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp2^model_2/decoder/conv2d_10/StatefulPartitionedCall2^model_2/decoder/conv2d_11/StatefulPartitionedCall2^model_2/decoder/conv2d_12/StatefulPartitionedCall2^model_2/decoder/conv2d_13/StatefulPartitionedCall2^model_2/decoder/conv2d_14/StatefulPartitionedCall2^model_2/decoder/conv2d_15/StatefulPartitionedCall2^model_2/decoder/conv2d_16/StatefulPartitionedCall2^model_2/decoder/conv2d_17/StatefulPartitionedCall1^model_2/decoder/conv2d_9/StatefulPartitionedCall4^model_2/encoder/block1_conv1/BiasAdd/ReadVariableOp6^model_2/encoder/block1_conv1/BiasAdd_1/ReadVariableOp3^model_2/encoder/block1_conv1/Conv2D/ReadVariableOp5^model_2/encoder/block1_conv1/Conv2D_1/ReadVariableOp4^model_2/encoder/block1_conv2/BiasAdd/ReadVariableOp6^model_2/encoder/block1_conv2/BiasAdd_1/ReadVariableOp3^model_2/encoder/block1_conv2/Conv2D/ReadVariableOp5^model_2/encoder/block1_conv2/Conv2D_1/ReadVariableOp4^model_2/encoder/block2_conv1/BiasAdd/ReadVariableOp6^model_2/encoder/block2_conv1/BiasAdd_1/ReadVariableOp3^model_2/encoder/block2_conv1/Conv2D/ReadVariableOp5^model_2/encoder/block2_conv1/Conv2D_1/ReadVariableOp4^model_2/encoder/block2_conv2/BiasAdd/ReadVariableOp6^model_2/encoder/block2_conv2/BiasAdd_1/ReadVariableOp3^model_2/encoder/block2_conv2/Conv2D/ReadVariableOp5^model_2/encoder/block2_conv2/Conv2D_1/ReadVariableOp4^model_2/encoder/block3_conv1/BiasAdd/ReadVariableOp6^model_2/encoder/block3_conv1/BiasAdd_1/ReadVariableOp3^model_2/encoder/block3_conv1/Conv2D/ReadVariableOp5^model_2/encoder/block3_conv1/Conv2D_1/ReadVariableOp4^model_2/encoder/block3_conv2/BiasAdd/ReadVariableOp6^model_2/encoder/block3_conv2/BiasAdd_1/ReadVariableOp3^model_2/encoder/block3_conv2/Conv2D/ReadVariableOp5^model_2/encoder/block3_conv2/Conv2D_1/ReadVariableOp4^model_2/encoder/block3_conv3/BiasAdd/ReadVariableOp6^model_2/encoder/block3_conv3/BiasAdd_1/ReadVariableOp3^model_2/encoder/block3_conv3/Conv2D/ReadVariableOp5^model_2/encoder/block3_conv3/Conv2D_1/ReadVariableOp4^model_2/encoder/block3_conv4/BiasAdd/ReadVariableOp6^model_2/encoder/block3_conv4/BiasAdd_1/ReadVariableOp3^model_2/encoder/block3_conv4/Conv2D/ReadVariableOp5^model_2/encoder/block3_conv4/Conv2D_1/ReadVariableOp4^model_2/encoder/block4_conv1/BiasAdd/ReadVariableOp6^model_2/encoder/block4_conv1/BiasAdd_1/ReadVariableOp3^model_2/encoder/block4_conv1/Conv2D/ReadVariableOp5^model_2/encoder/block4_conv1/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*£
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1model_2/decoder/conv2d_10/StatefulPartitionedCall1model_2/decoder/conv2d_10/StatefulPartitionedCall2f
1model_2/decoder/conv2d_11/StatefulPartitionedCall1model_2/decoder/conv2d_11/StatefulPartitionedCall2f
1model_2/decoder/conv2d_12/StatefulPartitionedCall1model_2/decoder/conv2d_12/StatefulPartitionedCall2f
1model_2/decoder/conv2d_13/StatefulPartitionedCall1model_2/decoder/conv2d_13/StatefulPartitionedCall2f
1model_2/decoder/conv2d_14/StatefulPartitionedCall1model_2/decoder/conv2d_14/StatefulPartitionedCall2f
1model_2/decoder/conv2d_15/StatefulPartitionedCall1model_2/decoder/conv2d_15/StatefulPartitionedCall2f
1model_2/decoder/conv2d_16/StatefulPartitionedCall1model_2/decoder/conv2d_16/StatefulPartitionedCall2f
1model_2/decoder/conv2d_17/StatefulPartitionedCall1model_2/decoder/conv2d_17/StatefulPartitionedCall2d
0model_2/decoder/conv2d_9/StatefulPartitionedCall0model_2/decoder/conv2d_9/StatefulPartitionedCall2j
3model_2/encoder/block1_conv1/BiasAdd/ReadVariableOp3model_2/encoder/block1_conv1/BiasAdd/ReadVariableOp2n
5model_2/encoder/block1_conv1/BiasAdd_1/ReadVariableOp5model_2/encoder/block1_conv1/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block1_conv1/Conv2D/ReadVariableOp2model_2/encoder/block1_conv1/Conv2D/ReadVariableOp2l
4model_2/encoder/block1_conv1/Conv2D_1/ReadVariableOp4model_2/encoder/block1_conv1/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block1_conv2/BiasAdd/ReadVariableOp3model_2/encoder/block1_conv2/BiasAdd/ReadVariableOp2n
5model_2/encoder/block1_conv2/BiasAdd_1/ReadVariableOp5model_2/encoder/block1_conv2/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block1_conv2/Conv2D/ReadVariableOp2model_2/encoder/block1_conv2/Conv2D/ReadVariableOp2l
4model_2/encoder/block1_conv2/Conv2D_1/ReadVariableOp4model_2/encoder/block1_conv2/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block2_conv1/BiasAdd/ReadVariableOp3model_2/encoder/block2_conv1/BiasAdd/ReadVariableOp2n
5model_2/encoder/block2_conv1/BiasAdd_1/ReadVariableOp5model_2/encoder/block2_conv1/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block2_conv1/Conv2D/ReadVariableOp2model_2/encoder/block2_conv1/Conv2D/ReadVariableOp2l
4model_2/encoder/block2_conv1/Conv2D_1/ReadVariableOp4model_2/encoder/block2_conv1/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block2_conv2/BiasAdd/ReadVariableOp3model_2/encoder/block2_conv2/BiasAdd/ReadVariableOp2n
5model_2/encoder/block2_conv2/BiasAdd_1/ReadVariableOp5model_2/encoder/block2_conv2/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block2_conv2/Conv2D/ReadVariableOp2model_2/encoder/block2_conv2/Conv2D/ReadVariableOp2l
4model_2/encoder/block2_conv2/Conv2D_1/ReadVariableOp4model_2/encoder/block2_conv2/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block3_conv1/BiasAdd/ReadVariableOp3model_2/encoder/block3_conv1/BiasAdd/ReadVariableOp2n
5model_2/encoder/block3_conv1/BiasAdd_1/ReadVariableOp5model_2/encoder/block3_conv1/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block3_conv1/Conv2D/ReadVariableOp2model_2/encoder/block3_conv1/Conv2D/ReadVariableOp2l
4model_2/encoder/block3_conv1/Conv2D_1/ReadVariableOp4model_2/encoder/block3_conv1/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block3_conv2/BiasAdd/ReadVariableOp3model_2/encoder/block3_conv2/BiasAdd/ReadVariableOp2n
5model_2/encoder/block3_conv2/BiasAdd_1/ReadVariableOp5model_2/encoder/block3_conv2/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block3_conv2/Conv2D/ReadVariableOp2model_2/encoder/block3_conv2/Conv2D/ReadVariableOp2l
4model_2/encoder/block3_conv2/Conv2D_1/ReadVariableOp4model_2/encoder/block3_conv2/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block3_conv3/BiasAdd/ReadVariableOp3model_2/encoder/block3_conv3/BiasAdd/ReadVariableOp2n
5model_2/encoder/block3_conv3/BiasAdd_1/ReadVariableOp5model_2/encoder/block3_conv3/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block3_conv3/Conv2D/ReadVariableOp2model_2/encoder/block3_conv3/Conv2D/ReadVariableOp2l
4model_2/encoder/block3_conv3/Conv2D_1/ReadVariableOp4model_2/encoder/block3_conv3/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block3_conv4/BiasAdd/ReadVariableOp3model_2/encoder/block3_conv4/BiasAdd/ReadVariableOp2n
5model_2/encoder/block3_conv4/BiasAdd_1/ReadVariableOp5model_2/encoder/block3_conv4/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block3_conv4/Conv2D/ReadVariableOp2model_2/encoder/block3_conv4/Conv2D/ReadVariableOp2l
4model_2/encoder/block3_conv4/Conv2D_1/ReadVariableOp4model_2/encoder/block3_conv4/Conv2D_1/ReadVariableOp2j
3model_2/encoder/block4_conv1/BiasAdd/ReadVariableOp3model_2/encoder/block4_conv1/BiasAdd/ReadVariableOp2n
5model_2/encoder/block4_conv1/BiasAdd_1/ReadVariableOp5model_2/encoder/block4_conv1/BiasAdd_1/ReadVariableOp2h
2model_2/encoder/block4_conv1/Conv2D/ReadVariableOp2model_2/encoder/block4_conv1/Conv2D/ReadVariableOp2l
4model_2/encoder/block4_conv1/Conv2D_1/ReadVariableOp4model_2/encoder/block4_conv1/Conv2D_1/ReadVariableOp:` \
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namecontent_image:^Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namestyle_image: 

_output_shapes
:: 

_output_shapes
:
é
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_8129

inputs
identity
MaxPoolMaxPoolinputs*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
r
IdentityIdentityMaxPool:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ
Ê
__inference_call_12188

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ""*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ä
f
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_12388

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
þ

F__inference_block4_conv1_layer_call_and_return_conditional_losses_8256

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

G__inference_block3_conv3_layer_call_and_return_conditional_losses_12093

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
¤
,__inference_block3_conv1_layer_call_fn_12042

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8182
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
e
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8897

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ñ
Ê
__inference_call_12358

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
³B
Á	
A__inference_encoder_layer_call_and_return_conditional_losses_8520

inputs+
block1_conv1_8468:@
block1_conv1_8470:@+
block1_conv2_8473:@@
block1_conv2_8475:@,
block2_conv1_8479:@ 
block2_conv1_8481:	-
block2_conv2_8484: 
block2_conv2_8486:	-
block3_conv1_8490: 
block3_conv1_8492:	-
block3_conv2_8495: 
block3_conv2_8497:	-
block3_conv3_8500: 
block3_conv3_8502:	-
block3_conv4_8505: 
block3_conv4_8507:	-
block4_conv1_8511: 
block4_conv1_8513:	
identity

identity_1

identity_2

identity_3¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_8468block1_conv1_8470*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_8102¾
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8473block1_conv2_8475*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_8119
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_8129¶
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_8479block2_conv1_8481*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8142¿
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8484block2_conv2_8486*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8159
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_8169¶
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_8490block3_conv1_8492*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8182¿
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8495block3_conv2_8497*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8199¿
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8500block3_conv3_8502*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8216¿
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8505block3_conv4_8507*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_8233
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_8243¶
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_8511block4_conv1_8513*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8256
IdentityIdentity-block1_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity-block2_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity-block3_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity-block4_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
¤
,__inference_block3_conv3_layer_call_fn_12082

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8216
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶B
Â	
A__inference_encoder_layer_call_and_return_conditional_losses_8722
input_2+
block1_conv1_8670:@
block1_conv1_8672:@+
block1_conv2_8675:@@
block1_conv2_8677:@,
block2_conv1_8681:@ 
block2_conv1_8683:	-
block2_conv2_8686: 
block2_conv2_8688:	-
block3_conv1_8692: 
block3_conv1_8694:	-
block3_conv2_8697: 
block3_conv2_8699:	-
block3_conv3_8702: 
block3_conv3_8704:	-
block3_conv4_8707: 
block3_conv4_8709:	-
block4_conv1_8713: 
block4_conv1_8715:	
identity

identity_1

identity_2

identity_3¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_8670block1_conv1_8672*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_8102¾
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8675block1_conv2_8677*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_8119
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_8129¶
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_8681block2_conv1_8683*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8142¿
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8686block2_conv2_8688*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8159
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_8169¶
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_8692block3_conv1_8694*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8182¿
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8697block3_conv2_8699*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8199¿
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8702block3_conv3_8704*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8216¿
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8707block3_conv4_8709*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_8233
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_8243¶
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_8713block4_conv1_8715*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8256
IdentityIdentity-block1_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity-block2_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity-block3_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity-block4_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ï
¦
'__inference_decoder_layer_call_fn_11791

inputs#
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	%

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@$

unknown_15:@

unknown_16:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_9223y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¶9
¼
A__inference_decoder_layer_call_and_return_conditional_losses_9223

inputs)
conv2d_9_9174:
conv2d_9_9176:	*
conv2d_10_9180:
conv2d_10_9182:	*
conv2d_11_9185:
conv2d_11_9187:	*
conv2d_12_9190:
conv2d_12_9192:	*
conv2d_13_9195:
conv2d_13_9197:	*
conv2d_14_9201:
conv2d_14_9203:	)
conv2d_15_9206:@
conv2d_15_9208:@(
conv2d_16_9212:@@
conv2d_16_9214:@(
conv2d_17_9217:@
conv2d_17_9219:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallö
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_9174conv2d_9_9176*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_8799ó
up_sampling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8812
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_10_9180conv2d_10_9182*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_8827
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_9185conv2d_11_9187*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_8846
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_9190conv2d_12_9192*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_8865
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_9195conv2d_13_9197*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_8884ö
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_8897
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_9201conv2d_14_9203*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_8912
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_9206conv2d_15_9208*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_8931õ
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8944
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_5/PartitionedCall:output:0conv2d_16_9212conv2d_16_9214*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_8959
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_9217conv2d_17_9219*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_8977
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

f
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_12380

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_12480

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
Ç
__inference_call_12556

inputs8
conv2d_readvariableop_resource:@)
add_readvariableop_resource:
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸
K
/__inference_up_sampling2d_5_layer_call_fn_12463

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_8776
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

×
'__inference_encoder_layer_call_fn_11290

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *Í
_output_shapesº
·:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_8520
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

G
+__inference_block2_pool_layer_call_fn_12023

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_8169{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

×
'__inference_encoder_layer_call_fn_11243

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *Í
_output_shapesº
·:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_8266
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

×
&__inference_encoder_layer_call_fn_8612
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	
identity

identity_1

identity_2

identity_3¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *Í
_output_shapesº
·:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_8520
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
¿
£
,__inference_block2_conv1_layer_call_fn_11982

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8142
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
À;
Ï
B__inference_decoder_layer_call_and_return_conditional_losses_11852

inputs*
conv2d_9_11794:
conv2d_9_11796:	+
conv2d_10_11803:
conv2d_10_11805:	+
conv2d_11_11808:
conv2d_11_11810:	+
conv2d_12_11813:
conv2d_12_11815:	+
conv2d_13_11818:
conv2d_13_11820:	+
conv2d_14_11827:
conv2d_14_11829:	*
conv2d_15_11832:@
conv2d_15_11834:@)
conv2d_16_11841:@@
conv2d_16_11843:@)
conv2d_17_11846:@
conv2d_17_11848:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢!conv2d_15/StatefulPartitionedCall¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallË
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_11794conv2d_9_11796*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7882f
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:Þ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor)conv2d_9/StatefulPartitionedCall:output:0up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0conv2d_10_11803conv2d_10_11805*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7904ò
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_11808conv2d_11_11810*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7922ò
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_11813conv2d_12_11815*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7940ò
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_11818conv2d_13_11820*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7958f
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   h
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:á
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor*conv2d_13/StatefulPartitionedCall:output:0up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0conv2d_14_11827conv2d_14_11829*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7980ó
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_11832conv2d_15_11834*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_7998f
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_5/mulMulup_sampling2d_5/Const:output:0 up_sampling2d_5/Const_1:output:0*
T0*
_output_shapes
:à
,up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor*conv2d_15/StatefulPartitionedCall:output:0up_sampling2d_5/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0conv2d_16_11841conv2d_16_11843*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8020ó
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_11846conv2d_17_11848*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_8037
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
°
G
+__inference_block3_pool_layer_call_fn_12118

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_8081
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
ø
D__inference_conv2d_11_layer_call_and_return_conditional_losses_12275

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ñ
 
(__inference_conv2d_9_layer_call_fn_12162

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_8799x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ÿ

G__inference_block3_conv4_layer_call_and_return_conditional_losses_12113

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
G
+__inference_block2_pool_layer_call_fn_12018

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_8069
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶B
Â	
A__inference_encoder_layer_call_and_return_conditional_losses_8667
input_2+
block1_conv1_8615:@
block1_conv1_8617:@+
block1_conv2_8620:@@
block1_conv2_8622:@,
block2_conv1_8626:@ 
block2_conv1_8628:	-
block2_conv2_8631: 
block2_conv2_8633:	-
block3_conv1_8637: 
block3_conv1_8639:	-
block3_conv2_8642: 
block3_conv2_8644:	-
block3_conv3_8647: 
block3_conv3_8649:	-
block3_conv4_8652: 
block3_conv4_8654:	-
block4_conv1_8658: 
block4_conv1_8660:	
identity

identity_1

identity_2

identity_3¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_8615block1_conv1_8617*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_8102¾
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8620block1_conv2_8622*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_8119
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_8129¶
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_8626block2_conv1_8628*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8142¿
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8631block2_conv2_8633*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8159
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_8169¶
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_8637block3_conv1_8639*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8182¿
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8642block3_conv2_8644*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8199¿
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8647block3_conv3_8649*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8216¿
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8652block3_conv4_8654*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_8233
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_8243¶
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_8658block4_conv1_8660*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8256
IdentityIdentity-block1_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

Identity_1Identity-block2_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_2Identity-block3_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Identity_3Identity-block4_conv1/StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall:j f
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

b
F__inference_block3_pool_layer_call_and_return_conditional_losses_12128

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8738

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
¡
)__inference_conv2d_12_layer_call_fn_12297

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_8865x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ð
É
__inference_call_7940

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿBB*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0§
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0t
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@P
ReluReluadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@s
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¡
ô
C__inference_conv2d_17_layer_call_and_return_conditional_losses_8977

inputs8
conv2d_readvariableop_resource:@)
add_readvariableop_resource:
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
mode	REFLECT|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0¨
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0u
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityadd:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ø
D__inference_conv2d_14_layer_call_and_return_conditional_losses_12410

inputs:
conv2d_readvariableop_resource:*
add_readvariableop_resource:	
identity¢Conv2D/ReadVariableOp¢add/ReadVariableOp
MirrorPad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
	MirrorPad	MirrorPadinputsMirrorPad/paddings:output:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
mode	REFLECT~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0©
Conv2DConv2DMirrorPad:output:0Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:*
dtype0v
addAddV2Conv2D:output:0add/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿR
ReluReluadd:z:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp^Conv2D/ReadVariableOp^add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿ: : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:Z V
2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¤
serving_default
Q
content_image@
serving_default_content_image:0ÿÿÿÿÿÿÿÿÿ
M
style_image>
serving_default_style_image:0ÿÿÿÿÿÿÿÿÿP
tf.clip_by_value_1:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ê
©
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer-9
layer_with_weights-1
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api

signatures
Î__call__
+Ï&call_and_return_all_conditional_losses
Ð_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
¿
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
 layer_with_weights-5
 layer-8
!layer_with_weights-6
!layer-9
"layer_with_weights-7
"layer-10
#layer-11
$layer_with_weights-8
$layer-12
%	variables
&trainable_variables
'regularization_losses
(	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"
_tf_keras_network
§
)	variables
*trainable_variables
+regularization_losses
,	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
´
-layer_with_weights-0
-layer-0
.layer-1
/layer_with_weights-1
/layer-2
0layer_with_weights-2
0layer-3
1layer_with_weights-3
1layer-4
2layer_with_weights-4
2layer-5
3layer-6
4layer_with_weights-5
4layer-7
5layer_with_weights-6
5layer-8
6layer-9
7layer_with_weights-7
7layer-10
8layer_with_weights-8
8layer-11
9	variables
:trainable_variables
;regularization_losses
<	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_sequential
(
=	keras_api"
_tf_keras_layer
¶
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
X26
Y27
Z28
[29
\30
]31
^32
_33
`34
a35"
trackable_list_wrapper
¶
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
P18
Q19
R20
S21
T22
U23
V24
W25
X26
Y27
Z28
[29
\30
]31
^32
_33
`34
a35"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
Î__call__
Ð_default_save_signature
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
-
×serving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_tf_keras_input_layer
½

>kernel
?bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"
_tf_keras_layer
½

@kernel
Abias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layer
§
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Bkernel
Cbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Dkernel
Ebias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
à__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer
§
{	variables
|trainable_variables
}regularization_losses
~	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"
_tf_keras_layer
À

Fkernel
Gbias
	variables
trainable_variables
regularization_losses
	keras_api
ä__call__
+å&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Hkernel
Ibias
	variables
trainable_variables
regularization_losses
	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Jkernel
Kbias
	variables
trainable_variables
regularization_losses
	keras_api
è__call__
+é&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Lkernel
Mbias
	variables
trainable_variables
regularization_losses
	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
ì__call__
+í&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Nkernel
Obias
	variables
trainable_variables
regularization_losses
	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"
_tf_keras_layer
¦
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17"
trackable_list_wrapper
¦
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
)	variables
*trainable_variables
+regularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
Ú

Pkernel
Pw
Qbias
Qb
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses
	òcall"
_tf_keras_layer
«
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú

Rkernel
Rw
Sbias
Sb
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses
	÷call"
_tf_keras_layer
Ú

Tkernel
Tw
Ubias
Ub
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses
	úcall"
_tf_keras_layer
Ú

Vkernel
Vw
Wbias
Wb
±	variables
²trainable_variables
³regularization_losses
´	keras_api
û__call__
+ü&call_and_return_all_conditional_losses
	ýcall"
_tf_keras_layer
Ú

Xkernel
Xw
Ybias
Yb
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses
	call"
_tf_keras_layer
«
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú

Zkernel
Zw
[bias
[b
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
__call__
+&call_and_return_all_conditional_losses
	call"
_tf_keras_layer
Ú

\kernel
\w
]bias
]b
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
__call__
+&call_and_return_all_conditional_losses
	call"
_tf_keras_layer
«
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú

^kernel
^w
_bias
_b
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
__call__
+&call_and_return_all_conditional_losses
	call"
_tf_keras_layer
Ú

`kernel
`w
abias
ab
Í	variables
Îtrainable_variables
Ïregularization_losses
Ð	keras_api
__call__
+&call_and_return_all_conditional_losses
	call"
_tf_keras_layer
¦
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
`16
a17"
trackable_list_wrapper
¦
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
`16
a17"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
9	variables
:trainable_variables
;regularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
/:-2block2_conv2/kernel
 :2block2_conv2/bias
/:-2block3_conv1/kernel
 :2block3_conv1/bias
/:-2block3_conv2/kernel
 :2block3_conv2/bias
/:-2block3_conv3/kernel
 :2block3_conv3/bias
/:-2block3_conv4/kernel
 :2block3_conv4/bias
/:-2block4_conv1/kernel
 :2block4_conv1/bias
+:)2conv2d_9/kernel
:2conv2d_9/bias
,:*2conv2d_10/kernel
:2conv2d_10/bias
,:*2conv2d_11/kernel
:2conv2d_11/bias
,:*2conv2d_12/kernel
:2conv2d_12/bias
,:*2conv2d_13/kernel
:2conv2d_13/bias
,:*2conv2d_14/kernel
:2conv2d_14/bias
+:)@2conv2d_15/kernel
:@2conv2d_15/bias
*:(@@2conv2d_16/kernel
:@2conv2d_16/bias
*:(@2conv2d_17/kernel
:2conv2d_17/bias
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
g	variables
htrainable_variables
iregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
o	variables
ptrainable_variables
qregularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
s	variables
ttrainable_variables
uregularization_losses
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
{	variables
|trainable_variables
}regularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
·
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¥	variables
¦trainable_variables
§regularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
©	variables
ªtrainable_variables
«regularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
­	variables
®trainable_variables
¯regularization_losses
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
±	variables
²trainable_variables
³regularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
µ	variables
¶trainable_variables
·regularization_losses
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
Í	variables
Îtrainable_variables
Ïregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
v
-0
.1
/2
03
14
25
36
47
58
69
710
811"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
é2æ
&__inference_model_2_layer_call_fn_9717
'__inference_model_2_layer_call_fn_10678
'__inference_model_2_layer_call_fn_10760
'__inference_model_2_layer_call_fn_10262À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_2_layer_call_and_return_conditional_losses_10978
B__inference_model_2_layer_call_and_return_conditional_losses_11196
B__inference_model_2_layer_call_and_return_conditional_losses_10387
B__inference_model_2_layer_call_and_return_conditional_losses_10512À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÝBÚ
__inference__wrapped_model_8048content_imagestyle_image"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
º2·
&__inference_encoder_layer_call_fn_8311
'__inference_encoder_layer_call_fn_11243
'__inference_encoder_layer_call_fn_11290
&__inference_encoder_layer_call_fn_8612
'__inference_encoder_layer_call_fn_11337
'__inference_encoder_layer_call_fn_11384À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
B__inference_encoder_layer_call_and_return_conditional_losses_11457
B__inference_encoder_layer_call_and_return_conditional_losses_11530
A__inference_encoder_layer_call_and_return_conditional_losses_8667
A__inference_encoder_layer_call_and_return_conditional_losses_8722
B__inference_encoder_layer_call_and_return_conditional_losses_11603
B__inference_encoder_layer_call_and_return_conditional_losses_11676À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_ada_in_1_layer_call_fn_11682¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_ada_in_1_layer_call_and_return_conditional_losses_11709¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
è2å
&__inference_decoder_layer_call_fn_9023
'__inference_decoder_layer_call_fn_11750
'__inference_decoder_layer_call_fn_11791
&__inference_decoder_layer_call_fn_9303À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
B__inference_decoder_layer_call_and_return_conditional_losses_11852
B__inference_decoder_layer_call_and_return_conditional_losses_11913
A__inference_decoder_layer_call_and_return_conditional_losses_9355
A__inference_decoder_layer_call_and_return_conditional_losses_9407À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÛBØ
#__inference_signature_wrapper_10596content_imagestyle_image"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block1_conv1_layer_call_fn_11922¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block1_conv1_layer_call_and_return_conditional_losses_11933¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block1_conv2_layer_call_fn_11942¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block1_conv2_layer_call_and_return_conditional_losses_11953¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
+__inference_block1_pool_layer_call_fn_11958
+__inference_block1_pool_layer_call_fn_11963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸2µ
F__inference_block1_pool_layer_call_and_return_conditional_losses_11968
F__inference_block1_pool_layer_call_and_return_conditional_losses_11973¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block2_conv1_layer_call_fn_11982¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block2_conv1_layer_call_and_return_conditional_losses_11993¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block2_conv2_layer_call_fn_12002¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block2_conv2_layer_call_and_return_conditional_losses_12013¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
+__inference_block2_pool_layer_call_fn_12018
+__inference_block2_pool_layer_call_fn_12023¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸2µ
F__inference_block2_pool_layer_call_and_return_conditional_losses_12028
F__inference_block2_pool_layer_call_and_return_conditional_losses_12033¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block3_conv1_layer_call_fn_12042¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block3_conv1_layer_call_and_return_conditional_losses_12053¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block3_conv2_layer_call_fn_12062¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block3_conv2_layer_call_and_return_conditional_losses_12073¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block3_conv3_layer_call_fn_12082¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block3_conv3_layer_call_and_return_conditional_losses_12093¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block3_conv4_layer_call_fn_12102¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block3_conv4_layer_call_and_return_conditional_losses_12113¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
+__inference_block3_pool_layer_call_fn_12118
+__inference_block3_pool_layer_call_fn_12123¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸2µ
F__inference_block3_pool_layer_call_and_return_conditional_losses_12128
F__inference_block3_pool_layer_call_and_return_conditional_losses_12133¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_block4_conv1_layer_call_fn_12142¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block4_conv1_layer_call_and_return_conditional_losses_12153¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2d_9_layer_call_fn_12162¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2d_9_layer_call_and_return_conditional_losses_12175¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12188¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_up_sampling2d_3_layer_call_fn_12193
/__inference_up_sampling2d_3_layer_call_fn_12198¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_12210
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_12218¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_10_layer_call_fn_12227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_10_layer_call_and_return_conditional_losses_12240¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12253¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_11_layer_call_fn_12262¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_11_layer_call_and_return_conditional_losses_12275¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12288¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_12_layer_call_fn_12297¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_12_layer_call_and_return_conditional_losses_12310¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12323¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_13_layer_call_fn_12332¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_13_layer_call_and_return_conditional_losses_12345¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12358¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_up_sampling2d_4_layer_call_fn_12363
/__inference_up_sampling2d_4_layer_call_fn_12368¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_12380
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_12388¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_14_layer_call_fn_12397¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_14_layer_call_and_return_conditional_losses_12410¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12423¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_15_layer_call_fn_12432¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_15_layer_call_and_return_conditional_losses_12445¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12458¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_up_sampling2d_5_layer_call_fn_12463
/__inference_up_sampling2d_5_layer_call_fn_12468¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_12480
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_12488¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_16_layer_call_fn_12497¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_16_layer_call_and_return_conditional_losses_12510¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12523¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_17_layer_call_fn_12532¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_17_layer_call_and_return_conditional_losses_12544¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_12556¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1
__inference__wrapped_model_8048õ(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`av¢s
l¢i
gd
1.
content_imageÿÿÿÿÿÿÿÿÿ
/,
style_imageÿÿÿÿÿÿÿÿÿ
ª "QªN
L
tf.clip_by_value_163
tf.clip_by_value_1ÿÿÿÿÿÿÿÿÿæ
C__inference_ada_in_1_layer_call_and_return_conditional_losses_11709l¢i
b¢_
]Z
+(
inputs/0ÿÿÿÿÿÿÿÿÿ  
+(
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 ¾
(__inference_ada_in_1_layer_call_fn_11682l¢i
b¢_
]Z
+(
inputs/0ÿÿÿÿÿÿÿÿÿ  
+(
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  Ü
G__inference_block1_conv1_layer_call_and_return_conditional_losses_11933>?I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ´
,__inference_block1_conv1_layer_call_fn_11922>?I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ü
G__inference_block1_conv2_layer_call_and_return_conditional_losses_11953@AI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ´
,__inference_block1_conv2_layer_call_fn_11942@AI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@é
F__inference_block1_pool_layer_call_and_return_conditional_losses_11968R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ×
F__inference_block1_pool_layer_call_and_return_conditional_losses_11973I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Á
+__inference_block1_pool_layer_call_fn_11958R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ®
+__inference_block1_pool_layer_call_fn_11963I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ý
G__inference_block2_conv1_layer_call_and_return_conditional_losses_11993BCI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
,__inference_block2_conv1_layer_call_fn_11982BCI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÞ
G__inference_block2_conv2_layer_call_and_return_conditional_losses_12013DEJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
,__inference_block2_conv2_layer_call_fn_12002DEJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
F__inference_block2_pool_layer_call_and_return_conditional_losses_12028R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ù
F__inference_block2_pool_layer_call_and_return_conditional_losses_12033J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_block2_pool_layer_call_fn_12018R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
+__inference_block2_pool_layer_call_fn_12023J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÞ
G__inference_block3_conv1_layer_call_and_return_conditional_losses_12053FGJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
,__inference_block3_conv1_layer_call_fn_12042FGJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÞ
G__inference_block3_conv2_layer_call_and_return_conditional_losses_12073HIJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
,__inference_block3_conv2_layer_call_fn_12062HIJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÞ
G__inference_block3_conv3_layer_call_and_return_conditional_losses_12093JKJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
,__inference_block3_conv3_layer_call_fn_12082JKJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÞ
G__inference_block3_conv4_layer_call_and_return_conditional_losses_12113LMJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
,__inference_block3_conv4_layer_call_fn_12102LMJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿé
F__inference_block3_pool_layer_call_and_return_conditional_losses_12128R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ù
F__inference_block3_pool_layer_call_and_return_conditional_losses_12133J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_block3_pool_layer_call_fn_12118R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
+__inference_block3_pool_layer_call_fn_12123J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÞ
G__inference_block4_conv1_layer_call_and_return_conditional_losses_12153NOJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
,__inference_block4_conv1_layer_call_fn_12142NOJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
__inference_call_12188aPQ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  {
__inference_call_12253aRS8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@{
__inference_call_12288aTU8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@{
__inference_call_12323aVW8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@{
__inference_call_12358aXY8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@
__inference_call_12423eZ[:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿ
ª "# ÿÿÿÿÿÿÿÿÿ~
__inference_call_12458d\]:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ@}
__inference_call_12523c^_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@}
__inference_call_12556c`a9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_10_layer_call_and_return_conditional_losses_12240nRS8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
)__inference_conv2d_10_layer_call_fn_12227aRS8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@¶
D__inference_conv2d_11_layer_call_and_return_conditional_losses_12275nTU8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
)__inference_conv2d_11_layer_call_fn_12262aTU8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@¶
D__inference_conv2d_12_layer_call_and_return_conditional_losses_12310nVW8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
)__inference_conv2d_12_layer_call_fn_12297aVW8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@¶
D__inference_conv2d_13_layer_call_and_return_conditional_losses_12345nXY8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
)__inference_conv2d_13_layer_call_fn_12332aXY8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@º
D__inference_conv2d_14_layer_call_and_return_conditional_losses_12410rZ[:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿ
ª "0¢-
&#
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_14_layer_call_fn_12397eZ[:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿ
ª "# ÿÿÿÿÿÿÿÿÿ¹
D__inference_conv2d_15_layer_call_and_return_conditional_losses_12445q\]:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv2d_15_layer_call_fn_12432d\]:¢7
0¢-
+(
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ@¸
D__inference_conv2d_16_layer_call_and_return_conditional_losses_12510p^_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv2d_16_layer_call_fn_12497c^_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@¸
D__inference_conv2d_17_layer_call_and_return_conditional_losses_12544p`a9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_17_layer_call_fn_12532c`a9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿµ
C__inference_conv2d_9_layer_call_and_return_conditional_losses_12175nPQ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 
(__inference_conv2d_9_layer_call_fn_12162aPQ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  Î
B__inference_decoder_layer_call_and_return_conditional_losses_11852PQRSTUVWXYZ[\]^_`a@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Î
B__inference_decoder_layer_call_and_return_conditional_losses_11913PQRSTUVWXYZ[\]^_`a@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Õ
A__inference_decoder_layer_call_and_return_conditional_losses_9355PQRSTUVWXYZ[\]^_`aH¢E
>¢;
1.
conv2d_9_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Õ
A__inference_decoder_layer_call_and_return_conditional_losses_9407PQRSTUVWXYZ[\]^_`aH¢E
>¢;
1.
conv2d_9_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¥
'__inference_decoder_layer_call_fn_11750zPQRSTUVWXYZ[\]^_`a@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¥
'__inference_decoder_layer_call_fn_11791zPQRSTUVWXYZ[\]^_`a@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª ""ÿÿÿÿÿÿÿÿÿ­
&__inference_decoder_layer_call_fn_9023PQRSTUVWXYZ[\]^_`aH¢E
>¢;
1.
conv2d_9_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ­
&__inference_decoder_layer_call_fn_9303PQRSTUVWXYZ[\]^_`aH¢E
>¢;
1.
conv2d_9_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª ""ÿÿÿÿÿÿÿÿÿ¨
B__inference_encoder_layer_call_and_return_conditional_losses_11457á>?@ABCDEFGHIJKLMNOQ¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "÷¢ó
ëç
74
0/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
85
0/1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¨
B__inference_encoder_layer_call_and_return_conditional_losses_11530á>?@ABCDEFGHIJKLMNOQ¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "÷¢ó
ëç
74
0/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
85
0/1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ô
B__inference_encoder_layer_call_and_return_conditional_losses_11603>?@ABCDEFGHIJKLMNOA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "³¢¯
§£
'$
0/0ÿÿÿÿÿÿÿÿÿ@
(%
0/1ÿÿÿÿÿÿÿÿÿ
&#
0/2ÿÿÿÿÿÿÿÿÿ@@
&#
0/3ÿÿÿÿÿÿÿÿÿ  
 Ô
B__inference_encoder_layer_call_and_return_conditional_losses_11676>?@ABCDEFGHIJKLMNOA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "³¢¯
§£
'$
0/0ÿÿÿÿÿÿÿÿÿ@
(%
0/1ÿÿÿÿÿÿÿÿÿ
&#
0/2ÿÿÿÿÿÿÿÿÿ@@
&#
0/3ÿÿÿÿÿÿÿÿÿ  
 ¨
A__inference_encoder_layer_call_and_return_conditional_losses_8667â>?@ABCDEFGHIJKLMNOR¢O
H¢E
;8
input_2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "÷¢ó
ëç
74
0/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
85
0/1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¨
A__inference_encoder_layer_call_and_return_conditional_losses_8722â>?@ABCDEFGHIJKLMNOR¢O
H¢E
;8
input_2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "÷¢ó
ëç
74
0/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
85
0/1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
85
0/3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ù
'__inference_encoder_layer_call_fn_11243Í>?@ABCDEFGHIJKLMNOQ¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ãß
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
63
1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
'__inference_encoder_layer_call_fn_11290Í>?@ABCDEFGHIJKLMNOQ¢N
G¢D
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ãß
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
63
1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
'__inference_encoder_layer_call_fn_11337ù>?@ABCDEFGHIJKLMNOA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "
%"
0ÿÿÿÿÿÿÿÿÿ@
&#
1ÿÿÿÿÿÿÿÿÿ
$!
2ÿÿÿÿÿÿÿÿÿ@@
$!
3ÿÿÿÿÿÿÿÿÿ  ¥
'__inference_encoder_layer_call_fn_11384ù>?@ABCDEFGHIJKLMNOA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "
%"
0ÿÿÿÿÿÿÿÿÿ@
&#
1ÿÿÿÿÿÿÿÿÿ
$!
2ÿÿÿÿÿÿÿÿÿ@@
$!
3ÿÿÿÿÿÿÿÿÿ  ù
&__inference_encoder_layer_call_fn_8311Î>?@ABCDEFGHIJKLMNOR¢O
H¢E
;8
input_2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ãß
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
63
1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
&__inference_encoder_layer_call_fn_8612Î>?@ABCDEFGHIJKLMNOR¢O
H¢E
;8
input_2+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ãß
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
63
1,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
2,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
63
3,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
B__inference_model_2_layer_call_and_return_conditional_losses_10387Û(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a~¢{
t¢q
gd
1.
content_imageÿÿÿÿÿÿÿÿÿ
/,
style_imageÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¢
B__inference_model_2_layer_call_and_return_conditional_losses_10512Û(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a~¢{
t¢q
gd
1.
content_imageÿÿÿÿÿÿÿÿÿ
/,
style_imageÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
B__inference_model_2_layer_call_and_return_conditional_losses_10978Ó(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`av¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
B__inference_model_2_layer_call_and_return_conditional_losses_11196Ó(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`av¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ú
'__inference_model_2_layer_call_fn_10262Î(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a~¢{
t¢q
gd
1.
content_imageÿÿÿÿÿÿÿÿÿ
/,
style_imageÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿò
'__inference_model_2_layer_call_fn_10678Æ(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`av¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿò
'__inference_model_2_layer_call_fn_10760Æ(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`av¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ
,)
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿù
&__inference_model_2_layer_call_fn_9717Î(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a~¢{
t¢q
gd
1.
content_imageÿÿÿÿÿÿÿÿÿ
/,
style_imageÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¼
#__inference_signature_wrapper_10596(>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a¢
¢ 
ª
B
content_image1.
content_imageÿÿÿÿÿÿÿÿÿ
>
style_image/,
style_imageÿÿÿÿÿÿÿÿÿ"QªN
L
tf.clip_by_value_163
tf.clip_by_value_1ÿÿÿÿÿÿÿÿÿí
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_12210R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_12218j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 Å
/__inference_up_sampling2d_3_layer_call_fn_12193R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
/__inference_up_sampling2d_3_layer_call_fn_12198]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ@@í
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_12380R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_12388l8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "0¢-
&#
0ÿÿÿÿÿÿÿÿÿ
 Å
/__inference_up_sampling2d_4_layer_call_fn_12363R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
/__inference_up_sampling2d_4_layer_call_fn_12368_8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "# ÿÿÿÿÿÿÿÿÿí
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_12480R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_12488l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Å
/__inference_up_sampling2d_5_layer_call_fn_12463R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
/__inference_up_sampling2d_5_layer_call_fn_12468_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@