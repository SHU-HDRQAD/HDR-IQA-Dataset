import torch
import timm
from efficientnet_pytorch import EfficientNet
from timm.models.vision_transformer import Block
from torch import nn
from einops import rearrange
from collections import OrderedDict
def get_all_model_devices(model):
    devices = []
    for mdl in model.state_dict().values():
        if mdl.device not in devices:
            devices.append(mdl.device)
    return devices


def load_checkpoint(my_model, path):
    load_net = torch.load(path, map_location=get_all_model_devices(my_model)[0])
    if 'state_dict' in load_net.keys():
        load_net = load_net['state_dict']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    my_model.load_state_dict(load_net_clean)

class attn(nn.Module):
    def __init__(self, channel, reduction=4, bias=True):
        super(attn, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_ca = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.SiLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

        self.conv_pa = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.BatchNorm2d(channel // reduction),
                nn.SiLU(inplace=True),
                nn.Conv2d(channel // reduction, channel // reduction, 3, padding=1, bias=bias),
                nn.BatchNorm2d(channel // reduction),
                nn.SiLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.BatchNorm2d(channel),
                nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_pool = self.avg_pool(x)
        x_ca = self.conv_ca(x_pool)
        x_pa = self.conv_pa(x)
        map = self.sigmoid(x_ca * x_pa)
        return map


class linear_patchfusion(nn.Module):
    def __init__(self, plane, reduction=4):
        super(linear_patchfusion, self).__init__()
        self.att1 = attn(plane, reduction=reduction)
        self.att2 = attn(plane, reduction=reduction)
        self.att3 = attn(plane, reduction=reduction)
        self.att4 = attn(plane, reduction=reduction)

    def forward(self, x):
        att1 = self.att1(x)
        z1 = att1 * x
        att2 = self.att2(z1)
        z2 = att2 * x
        att3 = self.att3(z2)
        z3 = att3 * x
        att4 = self.att4(z3)
        z = att4 * x + x
        return z

class CA_SA_module(nn.Module):
    def __init__(self, plane, reduction=4):
        super(CA_SA_module, self).__init__()
        self.exposure_error_mlp = nn.Sequential(
            nn.Conv2d(plane, plane // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(plane // reduction, plane, 1, 1, 0)
        )
        self.geometrical_mlp = nn.Sequential(
            nn.Conv2d(plane, plane // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(plane // reduction, plane, 1, 1, 0)
        )

        self.conv1 = nn.Conv2d(2, 1, 1, 1, 0)
        self.conv3 = nn.Conv2d(2, 1, 3, 1, 1)

        self.conv5 = nn.Conv2d(2, 1, 5, 1, 2)
        self.conv7 = nn.Conv2d(2, 1, 7, 1, 3)
        self.conv9 = nn.Sequential(
            nn.Conv2d(2, 2, 3, 1, 1),
            nn.Conv2d(2, 1, 3, 1, 1)
        )
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        exposure_error_att = self.exposure_error_mlp(self.avgpool(x))
        geometrical_att = self.geometrical_mlp(self.avgpool(x))

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        exposure_error_att = self.sigmoid(self.conv5(out) + self.conv7(out) + self.conv9(out) + exposure_error_att)
        geometrical_att = self.sigmoid(self.conv1(out) + self.conv3(out) + geometrical_att)
        x_exposure_error = exposure_error_att * x
        x_geometrical = geometrical_att * x

        x = x_exposure_error + x_geometrical + x

        return x


class linear_att(nn.Module):
    def __init__(self, plane, reduction=4):
        super(linear_att, self).__init__()
        self.att1 = CA_SA_module(plane, reduction)
        self.att2 = CA_SA_module(plane, reduction)
        self.att3 = CA_SA_module(plane, reduction)
        # self.att4 = CA_SA_module(plane, reduction)
    def forward(self, x):
        x = self.att1(x)
        x = self.att2(x)
        x = self.att3(x)
        # x = self.att4(x)
        return x
class qkv_patches_attention_module(nn.Module):
    def __init__(self, plane, size):
        super(qkv_patches_attention_module, self).__init__()
        self.conv_q = nn.Conv2d(plane, plane, 1, 1, 0)
        self.conv_k = nn.Conv2d(plane, plane, 1, 1, 0)
        self.conv_v = nn.Conv2d(plane, plane, 1, 1, 0)
        self.input_size = size

        self.q = nn.Linear(size ** 2, size ** 2)
        self.k = nn.Linear(size ** 2, size ** 2)
        self.v = nn.Linear(size ** 2, size ** 2)

        self.relu = nn.ReLU(inplace=True)
        self.norm = (size * size) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, patch1, patch2):
        # q = self.relu(self.conv_q(patch1))
        # q = rearrange(q, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        q = rearrange(patch1, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        q = self.q(q)

        # k = self.relu(self.conv_k(patch2))
        # k = rearrange(k, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        k = rearrange(patch2, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        k = self.k(k)

        attn = self.softmax(q @ k.transpose(-2, -1) * self.norm)

        # v = self.relu(self.conv_v(patch2))
        # v = rearrange(v, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        v = rearrange(patch2, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        v = self.v(v)

        out = attn @ v
        out = rearrange(out, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        return out

class patches_fusion_module(nn.Module):
    def __init__(self, plane, size):
        super(patches_fusion_module, self).__init__()
        self.att1 = qkv_patches_attention_module(plane, size)
        self.att2 = qkv_patches_attention_module(plane, size)
        self.patch_att1 = qkv_patches_attention_module(plane, size)
        self.patch_att2 = qkv_patches_attention_module(plane, size)
        self.conv = nn.Sequential(
            nn.Conv2d(plane * 2, plane, 1, 1, 0),
            nn.SiLU(inplace=True)
        )
    def forward(self, x1, x2):
        x1 = x1 + self.att1(x1, x1)
        x2 = x2 + self.att2(x2, x2)
        x1_ = self.patch_att1(x2, x1) + x1
        x2_ = self.patch_att2(x1, x2) + x2
        out = self.conv(torch.cat([x1_, x2_], dim=1))
        return out

class linear_enhance(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(linear_enhance, self).__init__()
        # self.cbam = CBAM(gate_channels=outchannel)
        self.conv1_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=outchannel * 3, out_channels=outchannel, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        x1 = self.conv1_1(x)
        # print('x1', x1.size())#[16, 24, 28, 28]
        x2 = self.conv3_1(x)
        # print('x2', x2.size())#[16, 24, 28, 28]
        x3 = self.conv3_2(x) + x1
        # print('x3', x3.size())#[16, 24, 28, 28]
        x3 = self.conv3_3(x3) + x2
        # print('x3', x3.size())#[16, 24, 28, 28]
        x = torch.cat((x1, x2, x3), 1)
        # print('x', x.size())#
        x = self.prelu(self.bn(self.conv1_2(x)))
        return x


class PixelCondition_PositionCoding(nn.Module):
    def __init__(self, inplane, outplane, embed_dim):
        super(PixelCondition_PositionCoding, self).__init__()
        self.PixelCondition = nn.Sequential(
            nn.Conv2d(inplane, 16, 4, 4,0),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, outplane, 4, 4, 0),
            nn.SiLU(inplace=True),
        )
        self.PositionCoding = nn.Parameter(torch.zeros(1, 1, embed_dim))
    def forward(self, x):
        PixelCondition = self.PixelCondition(x)
        PixelCondition = rearrange(PixelCondition, 'b c h w -> b (h w) c', h=14, w=14) # B, 196, 3
        PositionCoding = self.PositionCoding.expand(PixelCondition.size(0), -1, -1) # B, 1, 768
        return PixelCondition, PositionCoding


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=768, num_outputs=1, drop=0.1):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        del self.vit.blocks[11]
        del self.vit.blocks[10]
        del self.vit.blocks[9]
        del self.vit.blocks[8]
        self.save_output = SaveOutput()
        self.embed_dim = embed_dim
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        self.conv = nn.Sequential(
            nn.Linear(3072, embed_dim),
            nn.ReLU(inplace=True)
        )

        self.fusion1 = patches_fusion_module(embed_dim, 14)
        self.fusion2 = patches_fusion_module(embed_dim, 14)
        self.fusion3 = patches_fusion_module(embed_dim, 14)

        mymodel = EfficientNet.from_pretrained('efficientnet-b0', pretrained=True)
        mymodel._fc = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(1280, 500),
            nn.SiLU(inplace=True),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )
        load_checkpoint(mymodel, '/public/zzh/MANIQA my patch/parameter/Efficient_base/efficient_linear_base256_0.8406.pth')
        # load_checkpoint(mymodel, '/public/zzh/MANIQA my patch/parameter/Efficient_base/efficient_linear_base_0.8486.pth')
        mymodel.eval()
        features_linear = list(mymodel.children())
        self.model_layers_linear = nn.Sequential(*features_linear)
        self.conv_pool1 = nn.Sequential(
            # linear_enhance(24, 24),
            nn.Conv2d(24, 24, 3, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 24, 3, 2, 1),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(24),
        )
        self.conv_pool2 = nn.Sequential(
            # linear_enhance(40, 40),
            nn.Conv2d(40, 40, 3, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(40, 40, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(40),
        )
        self.conv_poo3 = nn.Sequential(
            # linear_enhance(112, 112),
            nn.Conv2d(112, 112, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.BatchNorm2d(112),
        )
        self.conv_pool4 = nn.Sequential(
            nn.Conv2d(1280, 320, 1, 1, 0),
            # linear_enhance(320, 320),
            nn.Conv2d(320, 320, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.BatchNorm2d(320),
        )

        self.DRAA = nn.Sequential(
            linear_att(496, 4),
            nn.Conv2d(496, 496, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.linear_patchfusion = nn.Sequential(
            nn.Conv2d(496 * 4, 496, 1, 1, 0),
            nn.ReLU(inplace=True),
            linear_patchfusion(496, 4),
        )
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim + 496, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim + 496, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

    def extract_feature(self, save_output):
        x6 = save_output.outputs[4][:, 1:]
        x7 = save_output.outputs[5][:, 1:]
        x8 = save_output.outputs[6][:, 1:]
        x9 = save_output.outputs[7][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def linear_features(self, x):
        f_linear = []
        with torch.no_grad():
            for i, model in enumerate(self.model_layers_linear):
                if i == 2:
                    for j, block in enumerate(model):
                        x = block(x)
                        if j == 2:
                            f_linear.append(x)  # j = 2, 4, 10,15
                        if j == 4:
                            f_linear.append(x)
                        if j == 10:
                            f_linear.append(x)
                elif i == 4:
                    x = model(x)
                    f_linear.append(x)
                else:
                    if i == 7:
                        x = x.view(x.size(0), -1)
                        x = model(x)
                    else:
                        x = model(x)
        # for i, model in enumerate(self.model_layers_linear):
        #     if i == 2:
        #         for j, block in enumerate(model):
        #             x = block(x)
        #             if j == 2:
        #                 f_linear.append(x)  # j = 2, 4, 10,15
        #             if j == 4:
        #                 f_linear.append(x)
        #             if j == 10:
        #                 f_linear.append(x)
        #     elif i == 4:
        #         x = model(x)
        #         f_linear.append(x)
        #     else:
        #         if i == 7:
        #             x = x.view(x.size(0), -1)
        #             x = model(x)
        #         else:
        #             x = model(x)
        f_linear[0] = self.conv_pool1(f_linear[0])
        f_linear[1] = self.conv_pool2(f_linear[1])
        f_linear[2] = self.conv_poo3(f_linear[2])
        f_linear[3] = self.conv_pool4(f_linear[3])
        return torch.cat([f_linear[0], f_linear[1], f_linear[2], f_linear[3]], dim=1)

    def forward(self, img_pu, img_linear):
        ''' perceptual network'''
        x1_ = torch.clone(img_pu)[:, 0:3, :, :]
        x2_ = torch.clone(img_pu)[:, 3:6, :, :]
        x3_ = torch.clone(img_pu)[:, 6:9, :, :]
        x4_ = torch.clone(img_pu)[:, 9:12, :, :]


        x1_ = self.vit(x1_)
        x1 = self.conv(self.extract_feature(self.save_output))
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=14, w=14)
        self.save_output.outputs.clear()

        x2_ = self.vit(x2_)
        x2 = self.conv(self.extract_feature(self.save_output))
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=14, w=14)
        self.save_output.outputs.clear()

        x3_ = self.vit(x3_)
        x3 = self.conv(self.extract_feature(self.save_output))
        x3 = rearrange(x3, 'b (h w) c -> b c h w', h=14, w=14)
        self.save_output.outputs.clear()

        x4_ = self.vit(x4_)
        x4 = self.conv(self.extract_feature(self.save_output))
        x4 = rearrange(x4, 'b (h w) c -> b c h w', h=14, w=14)
        self.save_output.outputs.clear()

        fusion1 = self.fusion1(x1, x2)
        fusion2 = self.fusion2(x3, x4)
        fusion = self.fusion3(fusion1, fusion2)

        ''' linear network'''

        x1 = torch.clone(img_linear)[:, 0:3, :, :]
        x2 = torch.clone(img_linear)[:, 3:6, :, :]
        x3 = torch.clone(img_linear)[:, 6:9, :, :]
        x4 = torch.clone(img_linear)[:, 9:12, :, :]

        x1 = self.linear_features(x1)
        x1 = self.DRAA(x1)

        x2 = self.linear_features(x2)
        x2 = self.DRAA(x2)

        x3 = self.linear_features(x3)
        x3 = self.DRAA(x3)

        x4 = self.linear_features(x4)
        x4 = self.DRAA(x4)

        linear_fusion = self.linear_patchfusion(torch.cat([x1, x2, x3, x4], dim=1))

        fusion = torch.cat([fusion, linear_fusion], dim=1)

        fusion = rearrange(fusion, 'b c h w -> b (h w) c', h=14, w=14)
        score = torch.tensor([]).cuda(img_pu.device)
        for i in range(img_pu.shape[0]):
            f = self.fc_score(fusion[i])
            w = self.fc_weight(fusion[i])
            _s = torch.sum(f * w) / (torch.sum(w) + 1e-10)
            score = torch.cat((score, _s.unsqueeze(0)), 0)

        return score

# from torchsummary import summary
# net = MANIQA(embed_dim=768).cuda('cuda:0')
# summary(net, ((12, 224, 224), (12, 224, 224)))
# a = torch.randn((1, 12, 224, 224)).cuda('cuda:0')
# out = net(a, a)
# print(out.size())