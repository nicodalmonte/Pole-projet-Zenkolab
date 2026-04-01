import timm, torch
m = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0)
ckpt = torch.load('cached_retfound/RETFound_dinov2_meh.pth', map_location='cpu')
print('Type of ckpt:', type(ckpt))
if 'model' in ckpt:
    ckpt = ckpt['model']
elif 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
print('Keys in ckpt:', list(ckpt.keys())[:5])
missing, unexpected = m.load_state_dict(ckpt, strict=False)
print('Missing:', len(missing), 'Unexpected:', len(unexpected))

