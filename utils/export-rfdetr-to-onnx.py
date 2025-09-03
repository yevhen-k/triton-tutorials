from rfdetr.detr import RFDETRMedium

model = RFDETRMedium()
# Exported graph: graph(%input :
# Float(1, 3, 576, 576, strides=[995328, 331776, 576, 1], requires_grad=0, device=cpu),
#
# PyTorch inference output shapes - Boxes:
# torch.Size([1, 3900, 4]), Labels: torch.Size([1, 3900, 91])
model.export()
