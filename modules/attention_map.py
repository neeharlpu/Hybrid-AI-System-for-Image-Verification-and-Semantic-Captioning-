import numpy as np
import cv2

def generate_gradcam(image, model, transform, device):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # Forward
    output = model(img_tensor)
    pred_class = output.argmax().item()

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    # ===== Grad-CAM++ =====
    grad_2 = grads ** 2
    grad_3 = grads ** 3
    sum_acts = np.sum(acts, axis=(1, 2), keepdims=True)

    alpha = grad_2 / (2 * grad_2 + sum_acts * grad_3 + 1e-8)
    weights = np.sum(alpha * np.maximum(grads, 0), axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    # ===== Normalize =====
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # ===== Smooth scaling =====
    cam = np.sqrt(cam)

    # ===== Resize =====
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)

    # ===== Smooth =====
    cam = cv2.GaussianBlur(cam, (25, 25), 0)

    # ===== Normalize again =====
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # 🔥 Focus boost (tightens attention)
    cam = cam ** 2

    # 🔥 Remove weak regions
    cam = np.where(cam > 0.3, cam, 0)

    # ===== Prepare image =====
    img_np = np.array(image.resize((224, 224)))

    # ===== Heatmap =====
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # ===== Overlay =====
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # ===== Highlight =====
    mask = cam > 0.6
    highlight = img_np.copy()
    highlight[~mask] = highlight[~mask] * 0.3

    return overlay, cam, highlight