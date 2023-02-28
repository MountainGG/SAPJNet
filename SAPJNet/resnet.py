from torch import nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock3d, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv3d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm3d(F1),
            nn.ReLU(True),
            nn.Conv3d(F1, F2, f, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(F2),
            nn.ReLU(True),
            nn.Conv3d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F3),
        )
        self.shortcut_1 = nn.Conv3d(
            in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm3d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ConvBlock2d(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock2d, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(
            in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock3d(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock3d, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv3d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F1),
            nn.ReLU(True),
            nn.Conv3d(F1, F2, f, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(F2),
            nn.ReLU(True),
            nn.Conv3d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock2d(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock2d, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ResModel3d(nn.Module):
    def __init__(self, channel):
        super(ResModel3d, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(channel[0], channel[1], 7,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(True),
        )
        self.stage2 = nn.Sequential(
            ConvBlock3d(channel[1], f=3, filters=[
                      channel[1], channel[1], channel[3]], s=2),
            IndentityBlock3d(channel[3], 3, [
                           channel[1], channel[1], channel[3]]),
            IndentityBlock3d(channel[3], 3, [
                           channel[1], channel[1], channel[3]]),
        )
        self.stage3 = nn.Sequential(
            ConvBlock3d(channel[3], f=3, filters=[
                      channel[2], channel[2], channel[4]], s=2),
            IndentityBlock3d(channel[4], 3, [
                           channel[2], channel[2], channel[4]]),
            IndentityBlock3d(channel[4], 3, [
                           channel[2], channel[2], channel[4]]),
            IndentityBlock3d(channel[4], 3, [
                           channel[2], channel[2], channel[4]]),
        )
        self.stage4 = nn.Sequential(
            ConvBlock3d(channel[4], f=3, filters=[
                      channel[3], channel[3], channel[5]], s=2),
            IndentityBlock3d(channel[5], 3, [
                           channel[3], channel[3], channel[5]]),
            IndentityBlock3d(channel[5], 3, [
                           channel[3], channel[3], channel[5]]),
            IndentityBlock3d(channel[5], 3, [
                           channel[3], channel[3], channel[5]]),
            IndentityBlock3d(channel[5], 3, [
                           channel[3], channel[3], channel[5]]),
            IndentityBlock3d(channel[5], 3, [
                           channel[3], channel[3], channel[5]]),
        )
        self.stage5 = nn.Sequential(
            ConvBlock3d(channel[5], f=3, filters=[
                      channel[4], channel[4], channel[6]], s=2),
            IndentityBlock3d(channel[6], 3, [
                           channel[4], channel[4], channel[6]]),
            IndentityBlock3d(channel[6], 3, [
                           channel[4], channel[4], channel[6]]),
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        return out


class ResModel2d(nn.Module):
    def __init__(self, channel):
        super(ResModel2d, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], 7,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(True),
        )
        self.stage2 = nn.Sequential(
            ConvBlock2d(channel[1], f=3, filters=[
                channel[1], channel[1], channel[3]], s=2),
            IndentityBlock2d(channel[3], 3, [
                channel[1], channel[1], channel[3]]),
            IndentityBlock2d(channel[3], 3, [
                channel[1], channel[1], channel[3]]),
        )
        self.stage3 = nn.Sequential(
            ConvBlock2d(channel[3], f=3, filters=[
                channel[2], channel[2], channel[4]], s=2),
            IndentityBlock2d(channel[4], 3, [
                channel[2], channel[2], channel[4]]),
            IndentityBlock2d(channel[4], 3, [
                channel[2], channel[2], channel[4]]),
            IndentityBlock2d(channel[4], 3, [
                channel[2], channel[2], channel[4]]),
        )
        self.stage4 = nn.Sequential(
            ConvBlock2d(channel[4], f=3, filters=[
                channel[3], channel[3], channel[5]], s=2),
            IndentityBlock2d(channel[5], 3, [
                channel[3], channel[3], channel[5]]),
            IndentityBlock2d(channel[5], 3, [
                channel[3], channel[3], channel[5]]),
            IndentityBlock2d(channel[5], 3, [
                channel[3], channel[3], channel[5]]),
            IndentityBlock2d(channel[5], 3, [
                channel[3], channel[3], channel[5]]),
            IndentityBlock2d(channel[5], 3, [
                channel[3], channel[3], channel[5]]),
        )
        self.stage5 = nn.Sequential(
            ConvBlock2d(channel[5], f=3, filters=[
                channel[4], channel[4], channel[6]], s=2),
            IndentityBlock2d(channel[6], 3, [
                channel[4], channel[4], channel[6]]),
            IndentityBlock2d(channel[6], 3, [
                channel[4], channel[4], channel[6]]),
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        return out
