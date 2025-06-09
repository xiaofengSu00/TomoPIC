import torch
import torch.nn as nn
import warnings
from model.Darknet import YOLOv5_C3CBAM,YOLO,YOLO_C2f
# from model.Darknet import YOLOv5_3D
class Detect_Framework(nn.Module):
    # YOLOv5 Detect head for detection models
    # scale_factor = 4  # strides computed during build
     # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    scale_factor = 4

    def __init__(self, class_num=12, anchors=(),anchor_num=3,inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.class_num = class_num  # number of classes  80
        self.out_channels = class_num + 5  # number of outputs per anchor 85

        self.anchor = anchors
        self.anchor_num = anchor_num

        # self.network = YOLO_C2f(in_channels=1,out_channels=self.anchor_num * self.out_channels,base_channels=32)
        # self.network = YOLO(in_channels=1,out_channels=self.anchor_num * self.out_channels,base_channels=32)
        self.network = YOLOv5_C3CBAM(in_channels=1,out_channels=self.anchor_num * self.out_channels,base_channels=32)

        self.grid = torch.empty(0)  # init grid  list [tensor([]),tensor([]),tensor([])]
        self.anchor_grid = torch.empty(0)   # init anchor grid
        # self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.export = False

        self.scale_factor = 4

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        # z = []  # inference output

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor


        # out1,out2,out3 = self.network(x)
        out3 = self.network(x)

        batch_size,_,gz,gy,gx = out3.shape

        tmp = out3.view(batch_size,self.anchor_num,self.out_channels,gz,gy,gx).permute(0,1,3,4,5,2).contiguous()

        if not self.training:
            self.grid,self.anchor_grid = self._make_grid(tmp,gz,gy,gx)
            xyz,r,conf = tmp.sigmoid().split((3,1,self.class_num+1),5)

            xyz = ((xyz *2) + self.grid) * self.scale_factor
            r = (r*2)**2 * self.anchor_grid
            pred = torch.cat((xyz,r,conf),-1)

        return tmp if self.training else (tmp,pred)


    def _make_grid(self,p,gz,gy,gx,):
        FloatTensor = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        grid_shape = 1,self.anchor_num,gz,gy,gx,3
        anchor_grid_shape = 1,self.anchor_num,gz,gy,gx,1

        grid_z = torch.arange(gz).type(FloatTensor)
        grid_y = torch.arange(gy).type(FloatTensor)
        grid_x = torch.arange(gx).type(FloatTensor)

        zv,yv,xv = torch.meshgrid(grid_z,grid_y,grid_x)
       
        grid = (torch.stack((xv,yv,zv),3).expand(grid_shape) - 0.5).type(FloatTensor) ## (1,3,16,16,16,3)

        anchor_grid = (self.anchor[0,...].view(1,self.anchor_num,1,1,1,1).expand(anchor_grid_shape)).type(FloatTensor) ## (1,3,16,16,16,1)

        return grid,anchor_grid



if __name__ == '__main__':
    imgs = torch.randn(size=(2,1,64,64,64))
    anchors = torch.tensor([[[5],[10],[15]]])
    model = Detect_Framework(anchors = anchors)
    model.train()
    pred = model(imgs)
    print('-----------')