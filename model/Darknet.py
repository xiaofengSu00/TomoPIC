import warnings
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
	default_act = nn.SiLU()
	def __init__(self,c1,c2,k,s,p=None,g=1,d=1,act=True):
		super().__init__()

		pad = autopad(k,p,d)
		self.conv = nn.Conv3d(in_channels=c1, out_channels=c2, kernel_size=int(k),stride=s,padding=pad , groups=g, dilation=d, bias=False)
		self.bn = nn.BatchNorm3d(c2)
		self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

	def forward(self, x): ## 2,64,32,32,32
		"""Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
		return self.act(self.bn(self.conv(x)))  ## CBA

	def forward_fuse(self, x):
		"""Applies a fused convolution and activation function to the input tensor `x`."""
		return self.act(self.conv(x))

class Bottleneck(nn.Module):   ### residual block --- if shortcut 2*conv + x else 2*conv
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1,1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=3):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool3d(kernel_size=(k,k,k), stride=(1,1,1), padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Concat(nn.Module):  ### concat by special demension
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)

class YOLO(nn.Module):
    def __init__(self,in_channels,out_channels,base_channels=32):
        super().__init__()
        self.conv_0 = Conv(in_channels,base_channels,3,1,1)
        
        self.conv_1 = Conv(base_channels,base_channels*2,3,2,1)  ## downsample
        self.CSP_2 = C3(base_channels*2,base_channels*2,n=3,shortcut=True)
        
        self.conv_3 = Conv(base_channels*2,base_channels*4,3,2,p=1)
        self.CSP_4 = C3(base_channels*4,base_channels*4,n = 6 ,shortcut=True)
        
        self.conv_5 = Conv(base_channels*4,base_channels*8,k=3,s=2,p=1)
        self.CSP_6 = C3(base_channels*8,base_channels*8,n=9,shortcut=True)
        
        self.conv_7 = Conv(base_channels*8,base_channels*16,k=3,s=2,p=1)
        self.CSP_8 = C3(base_channels*16,base_channels*16,n=3,shortcut=True)
        
        self.SPPF_9 = SPPF(base_channels*16,base_channels*16,k=3) #5
        
		### head
        self.conv_10 = Conv(base_channels*16,base_channels*8,1,1)
        self.up_11 = nn.Upsample(scale_factor=2)
        self.cat_12 = Concat()
        self.CSP2_13 = C3(base_channels*8*2,base_channels*8,n=3,shortcut=False)
        
        self.conv_14 = Conv(base_channels*8,base_channels*4,1,1)
        self.up_15 = nn.Upsample(scale_factor=2)
        self.cat_16 = Concat()
        self.CSP2_17 = C3(base_channels*4*2,base_channels*4,n=3,shortcut=False)
        
        self.detection3 = nn.Conv3d(in_channels=128,out_channels=out_channels,kernel_size=1)
    
    def forward(self,x):
        x = self.conv_0(x)  
       
        x = self.conv_1(x)  
        x = self.CSP_2(x)   
        
        x = self.conv_3(x) 
        x = self.CSP_4(x)   
        x1 = x

        x = self.conv_5(x)  
        x = self.CSP_6(x)   
        x2 = x
		
        x = self.conv_7(x)  
        x = self.CSP_8(x)   
        
        # x = self.CBAM(x)
        x = self.SPPF_9(x)  
        x = self.conv_10(x)  
		# x3 = x
        
        x = self.up_11(x)   ## 
        x = self.cat_12((x2,x))  ## 
        x = self.CSP2_13(x)  ## 
        x = self.conv_14(x)  ##
		# x4 = x
        
        x = self.up_15(x)  ## up 
        x =self.cat_16((x1,x)) ## 
        x = self.CSP2_17(x)  ## 
        out3 = self.detection3(x)  ## 

        return out3 

class ChannelAttention(nn.Module):
    def __init__(self,in_channels,ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.f1 = nn.Conv3d(in_channels,in_channels//ratio,1,bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv3d(in_channels//ratio,in_channels,1,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out * max_out)

        return out

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=3):
        super().__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv3d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv(x)

        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self,c1,c2,ratio=16,kernel_size=3):
        super().__init__()
        self.channel = ChannelAttention(c1,ratio)
        self.spatial = SpatialAttention(kernel_size)
    
    def forward(self,x):
        out = self.channel(x) * x
        out = self.spatial(out) * out
        return out

class CBAMBottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5,ratio=16,kernel_size=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

        self.channel = ChannelAttention(c2,ratio)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        x2 = self.cv2(self.cv1(x))
        out = self.channel(x2) * x2
        out = self.spatial(out) * out
        
        return x + out if self.add else out

class C3CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1,1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class YOLOv5_Add_CBAM(nn.Module):
    def __init__(self,in_channels,out_channels,base_channels=32):
        super().__init__()
        self.conv_0 = Conv(in_channels,base_channels,3,1,1)
        
        self.conv_1 = Conv(base_channels,base_channels*2,3,2,1)  ## downsample
        self.CSP_2 = C3(base_channels*2,base_channels*2,n=3)
        
        self.conv_3 = Conv(base_channels*2,base_channels*4,3,2,p=1)
        self.CSP_4 = C3(base_channels*4,base_channels*4,n =6)
        
        self.conv_5 = Conv(base_channels*4,base_channels*8,k=3,s=2,p=1)
        self.CSP_6 = C3(base_channels*8,base_channels*8,n=9)
        
        self.conv_7 = Conv(base_channels*8,base_channels*16,k=3,s=2,p=1)
        self.CSP_8 = C3(base_channels*16,base_channels*16,n=3)
        
        self.SPPF_9 = SPPF(base_channels*16,base_channels*16,k=3) #5
        
		### head
        self.conv_10 = Conv(base_channels*16,base_channels*8,1,1)
        self.up_11 = nn.Upsample(scale_factor=2)
        self.cat_12 = Concat()
        self.CSP2_13 = C3(base_channels*8*2,base_channels*8,n=3,shortcut=False)
        
        self.conv_14 = Conv(base_channels*8,base_channels*4,1,1)
        self.up_15 = nn.Upsample(scale_factor=2)
        self.cat_16 = Concat()
        self.CSP2_17 = C3(base_channels*4*2,base_channels*4,n=3,shortcut=False)
        self.CBAM = CBAM(base_channels*16,base_channels*16)  
        self.detection3 = nn.Conv3d(in_channels=base_channels*4,out_channels=out_channels,kernel_size=1)
        
    def forward(self,x):
        x = self.conv_0(x)  ## (bs,32,64,64,64)
       
        x = self.conv_1(x)  ## (bs,64,32,32,32)
        x = self.CSP_2(x)   ## (bs,64,32,32,32)
        
        x = self.conv_3(x)  ## (bs,128,16,16,16)
        x = self.CSP_4(x)   ## (bs,128,16,16,16)
        x1 = x

        x = self.conv_5(x)  ## (bs,256,8,8,8)
        x = self.CSP_6(x)   ## (bs,256,8,8,8)
        x2 = x
		
        x = self.conv_7(x)  ## (bs,512,4,4,4)
        x = self.CSP_8(x)   ## (bs,512,4,4,4)
        
        x = self.CBAM(x)

        x = self.SPPF_9(x)   ## (bs,512,4,4,4)
        x = self.conv_10(x)  ## (bs,256,4,4,4)
		# x3 = x
        
        x = self.up_11(x)   ## (bs,256,8,8,8)
        x = self.cat_12((x2,x))  ## (bs,512,8,8,8)
        x = self.CSP2_13(x)  ## (bs,256,8,8,8)
        x = self.conv_14(x)  ## (bs,128,8,8,8)
		# x4 = x
        
        x = self.up_15(x)  ## up (bs,128,16,16,16)
        x =self.cat_16((x1,x)) ## (bs,256,16,16,16)
        x = self.CSP2_17(x)  ## (bs,128,16,16,16)
        out3 = self.detection3(x)  ## 
        
        return out3 #out1,out2,out3

class YOLOv5_C3CBAM(nn.Module):
    def __init__(self,in_channels,out_channels,base_channels=32):
        super().__init__()
        self.conv_0 = Conv(in_channels,base_channels,3,1,1)
        
        self.conv_1 = Conv(base_channels,base_channels*2,3,2,1)  ## downsample
        self.CSP_2 = C3CBAM(base_channels*2,base_channels*2,n=3)
        
        self.conv_3 = Conv(base_channels*2,base_channels*4,3,2,p=1)
        self.CSP_4 = C3CBAM(base_channels*4,base_channels*4,n =6)
        
        self.conv_5 = Conv(base_channels*4,base_channels*8,k=3,s=2,p=1)
        self.CSP_6 = C3CBAM(base_channels*8,base_channels*8,n=9)
        
        self.conv_7 = Conv(base_channels*8,base_channels*16,k=3,s=2,p=1)
        self.CSP_8 = C3CBAM(base_channels*16,base_channels*16,n=3)
        
        self.SPPF_9 = SPPF(base_channels*16,base_channels*16,k=3) #5
        
		### head
        self.conv_10 = Conv(base_channels*16,base_channels*8,1,1)
        self.up_11 = nn.Upsample(scale_factor=2)
        self.cat_12 = Concat()
        self.CSP2_13 = C3CBAM(base_channels*8*2,base_channels*8,n=3,shortcut=False)
        
        self.conv_14 = Conv(base_channels*8,base_channels*4,1,1)
        self.up_15 = nn.Upsample(scale_factor=2)
        self.cat_16 = Concat()
        self.CSP2_17 = C3CBAM(base_channels*4*2,base_channels*4,n=3,shortcut=False)
       
        self.detection3 = nn.Conv3d(in_channels=base_channels*4,out_channels=out_channels,kernel_size=1)
        
    def forward(self,x):
        x = self.conv_0(x)  ## (bs,32,64,64,64)
       
        x = self.conv_1(x)  ## (bs,64,32,32,32)
        x = self.CSP_2(x)   ## (bs,64,32,32,32)
        
        x = self.conv_3(x)  ## (bs,128,16,16,16)
        x = self.CSP_4(x)   ## (bs,128,16,16,16)
        x1 = x

        x = self.conv_5(x)  ## (bs,256,8,8,8)
        x = self.CSP_6(x)   ## (bs,256,8,8,8)
        x2 = x
		
        x = self.conv_7(x)  ## (bs,512,4,4,4)
        x = self.CSP_8(x)   ## (bs,512,4,4,4)

        x = self.SPPF_9(x)   ## (bs,512,4,4,4)
        x = self.conv_10(x)  ## (bs,256,4,4,4)
		# x3 = x
        
        x = self.up_11(x)   ## (bs,256,8,8,8)
        x = self.cat_12((x2,x))  ## (bs,512,8,8,8)
        x = self.CSP2_13(x)  ## (bs,256,8,8,8)
        x = self.conv_14(x)  ## (bs,128,8,8,8)
		# x4 = x
        
        x = self.up_15(x)  ## up (bs,128,16,16,16)
        x =self.cat_16((x1,x)) ## (bs,256,16,16,16)
        x = self.CSP2_17(x)  ## (bs,128,16,16,16)
        out3 = self.detection3(x)  ## 
        
        return out3 #out1,out2,out3


                        
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, z_dim, y_dim, x_dim = input_tensor.size()
        x_range = torch.linspace(-1,1,x_dim)
        y_range = torch.linspace(-1,1,y_dim)
        z_range = torch.linspace(-1,1,z_dim)

        zz_channel, yy_channel, xx_channel = torch.meshgrid(z_range,y_range,x_range)

        xx_channel = xx_channel.expand([batch_size, 1, -1, -1, -1])
        yy_channel = yy_channel.expand([batch_size, 1, -1, -1, -1])
        zz_channel = zz_channel.expand([batch_size, 1, -1, -1, -1])

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor),
            zz_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 3
        if with_r:
            in_channels += 1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x
    

class YOLOv5_Add_Coordconv(nn.Module):
    def __init__(self,in_channels,out_channels,base_channels=32):
        super().__init__()
        self.conv_0 = Conv(in_channels,base_channels,3,1,1)
        
        self.conv_1 = Conv(base_channels,base_channels*2,3,2,1)  ## downsample
        self.CSP_2 = C3CBAM(base_channels*2,base_channels*2,n=3)
        
        self.conv_3 = Conv(base_channels*2,base_channels*4,3,2,p=1)
        self.CSP_4 = C3CBAM(base_channels*4,base_channels*4,n =6)
        
        self.conv_5 = Conv(base_channels*4,base_channels*8,k=3,s=2,p=1)
        self.CSP_6 = C3CBAM(base_channels*8,base_channels*8,n=9)
        
        self.conv_7 = Conv(base_channels*8,base_channels*16,k=3,s=2,p=1)
        self.CSP_8 = C3CBAM(base_channels*16,base_channels*16,n=3)
        
        self.SPPF_9 = SPPF(base_channels*16,base_channels*16,k=3) #5
        
		### head
        self.conv_10 = CoordConv(base_channels*16,base_channels*8,1,1)
        # self.conv_10 = Conv(base_channels*16,base_channels*8,1,1)
        self.up_11 = nn.Upsample(scale_factor=2)
        self.cat_12 = Concat()
        self.CSP2_13 = C3CBAM(base_channels*8*2,base_channels*8,n=3,shortcut=False)
        
        self.conv_14 = CoordConv(base_channels*8,base_channels*4,1,1)
        # self.conv_14 = Conv(base_channels*8,base_channels*4,1,1)
        self.up_15 = nn.Upsample(scale_factor=2)
        self.cat_16 = Concat()
        self.CSP2_17 = C3CBAM(base_channels*4*2,base_channels*4,n=3,shortcut=False)
        # self.CBAM = CBAMLayer(base_channels*16,spatial_kernel=3)  
        self.detection3 = nn.Conv3d(in_channels=base_channels*4,out_channels=out_channels,kernel_size=1)
        
    def forward(self,x):
        x = self.conv_0(x)  ## (bs,32,64,64,64)
       
        x = self.conv_1(x)  ## (bs,64,32,32,32)
        x = self.CSP_2(x)   ## (bs,64,32,32,32)
        
        x = self.conv_3(x)  ## (bs,128,16,16,16)
        x = self.CSP_4(x)   ## (bs,128,16,16,16)
        x1 = x

        x = self.conv_5(x)  ## (bs,256,8,8,8)
        x = self.CSP_6(x)   ## (bs,256,8,8,8)
        x2 = x
		
        x = self.conv_7(x)  ## (bs,512,4,4,4)
        x = self.CSP_8(x)   ## (bs,512,4,4,4

        x = self.SPPF_9(x)   ## (bs,512,4,4,4)
        x = self.conv_10(x)  ## (bs,256,4,4,4)
		# x3 = x
        
        x = self.up_11(x)   ## (bs,256,8,8,8)
        x = self.cat_12((x2,x))  ## (bs,512,8,8,8)
        x = self.CSP2_13(x)  ## (bs,256,8,8,8)
        x = self.conv_14(x)  ## (bs,128,8,8,8)
		# x4 = x
        
        x = self.up_15(x)  ## up (bs,128,16,16,16)
        x =self.cat_16((x1,x)) ## (bs,256,16,16,16)
        x = self.CSP2_17(x)  ## (bs,128,16,16,16)
        out3 = self.detection3(x)  ## 
        
        return out3 #out1,out2,out3  
           
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class YOLO_C2f(nn.Module):
    def __init__(self,in_channels,out_channels,base_channels=32):
        super().__init__()
        self.conv_0 = Conv(in_channels,base_channels,3,1,1)
        
        self.conv_1 = Conv(base_channels,base_channels*2,3,2,1)  ## downsample
        self.CSP_2 = C2f(base_channels*2,base_channels*2,n=3,shortcut=True)
        
        self.conv_3 = Conv(base_channels*2,base_channels*4,3,2,p=1)
        self.CSP_4 = C2f(base_channels*4,base_channels*4,n = 6 ,shortcut=True)
        
        self.conv_5 = Conv(base_channels*4,base_channels*8,k=3,s=2,p=1)
        self.CSP_6 = C2f(base_channels*8,base_channels*8,n=9,shortcut=True)
        
        self.conv_7 = Conv(base_channels*8,base_channels*16,k=3,s=2,p=1)
        self.CSP_8 = C2f(base_channels*16,base_channels*16,n=3,shortcut=True)
        
        self.SPPF_9 = SPPF(base_channels*16,base_channels*16,k=3) #5
        
		### head
        self.conv_10 = Conv(base_channels*16,base_channels*8,1,1)
        self.up_11 = nn.Upsample(scale_factor=2)
        self.cat_12 = Concat()
        self.CSP2_13 = C2f(base_channels*8*2,base_channels*8,n=3,shortcut=False)
        
        self.conv_14 = Conv(base_channels*8,base_channels*4,1,1)
        self.up_15 = nn.Upsample(scale_factor=2)
        self.cat_16 = Concat()
        self.CSP2_17 = C2f(base_channels*4*2,base_channels*4,n=3,shortcut=False)
        
        self.detection3 = nn.Conv3d(in_channels=128,out_channels=out_channels,kernel_size=1)
    
    def forward(self,x):
        x = self.conv_0(x)  
       
        x = self.conv_1(x)  
        x = self.CSP_2(x)   
        
        x = self.conv_3(x) 
        x = self.CSP_4(x)   
        x1 = x

        x = self.conv_5(x)  
        x = self.CSP_6(x)   
        x2 = x
		
        x = self.conv_7(x)  
        x = self.CSP_8(x)   
        
        # x = self.CBAM(x)
        x = self.SPPF_9(x)  
        x = self.conv_10(x)  
		# x3 = x
        
        x = self.up_11(x)   ## 
        x = self.cat_12((x2,x))  ## 
        x = self.CSP2_13(x)  ## 
        x = self.conv_14(x)  ##
		# x4 = x
        
        x = self.up_15(x)  ## up 
        x =self.cat_16((x1,x)) ## 
        x = self.CSP2_17(x)  ## 
        out3 = self.detection3(x)  ## 

        return out3 

if __name__ == '__main__':
    ### 添加小目标检测头
    ### C3---> C2f
    ### bifpn
    x = torch.randn(size=(2, 1, 64, 64, 64))
    model = YOLO(in_channels=1,out_channels=51,base_channels=32)
    out = model(x)

    print('-------------')