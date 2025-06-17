# SECTION 1: CNN FROM SCRATCH (NumPy implementation)

np.random.seed(1) # For reproducibility

# ---------- ZERO PADDING ----------
def zero_pad(A, pad):
    A_padded = np.pad(A, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=(0, 0))
    return A_padded

# ---------- ONE STEP CONVOLUTION ----------
def one_step_conv(region, filter, bias):
    elementwise_product = region * filter
    summed_values = np.sum(elementwise_product)
    result = summed_values + bias
    return result.item()

# ---------- GET REGION FOR CONV ----------
def get_region(A_padded, i, j, stride, filter_size):
    vert_start = i * stride
    vert_end = vert_start + filter_size
    horiz_start = j * stride
    horiz_end = horiz_start + filter_size
    region = A_padded[vert_start:vert_end, horiz_start:horiz_end]
    return region

# ---------- ReLU ACTIVATION ----------
class ReLU:
    def __init__(self):
        pass
    def forward(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return np.where(x >= 0, 1, 0)
    def __call__(self, x):
        return self.forward(x)

# ---------- CONVOLUTION FORWARD ----------
def conv_forward(A_prev, filter, bias, stride=1, pad=0, act=ReLU()):
    batch_size = A_prev.shape[0]
    n = A_prev.shape[1]
    f = filter.shape[0]
    m = (n + 2 * pad - f) // stride + 1
    Z = np.zeros((batch_size, m, m))
    A_prev_padded = zero_pad(A_prev, pad)
    for b in range(batch_size):
        for i in range(m):
            for j in range(m):
                region = get_region(A_prev_padded[b], i, j, stride, f)
                Z[b, i, j] = one_step_conv(region, filter, bias)
    A = act(Z)
    cache = (A_prev, Z)
    return A, cache

# ---------- CONVOLUTION BACKWARD ----------
def conv_backward(dA, filter, bias, cache, act=ReLU(), stride=1, pad=0):
    A_prev, Z = cache
    batch_size = A_prev.shape[0]
    n = A_prev.shape[1]
    f = filter.shape[0]
    m = (n + 2 * pad - f) // stride + 1
    dF = np.zeros_like(filter, dtype=np.float64)
    db = np.zeros_like(bias, dtype=np.float64)
    dA_prev = np.zeros_like(A_prev, dtype=np.float64)
    if pad > 0:
        A_prev_padded = zero_pad(A_prev, pad)
        dA_prev_padded = zero_pad(dA_prev, pad)
    else:
        A_prev_padded = A_prev
        dA_prev_padded = dA_prev
    dZ = act.derivative(Z) * dA
    for b in range(batch_size):
        for i in range(m):
            for j in range(m):
                region = get_region(A_prev_padded[b], i, j, stride, f)
                dF += dZ[b, i, j] * region
                db += dZ[b, i, j]
                dA_prev_padded[b, i*stride:i*stride+f, j*stride:j*stride+f] += filter * dZ[b, i, j]
    if pad > 0:
        dA_prev = dA_prev_padded[:, pad:-pad, pad:-pad]
    else:
        dA_prev = dA_prev_padded
    return dA_prev, dF, db

# ---------- FLATTEN ----------
def flatten(A):
    batch_size = A.shape[0]
    m = A.shape[1]
    A_flattened = A.reshape(batch_size, m * m)
    return A_flattened

# ---------- FULLY CONNECTED LAYER ----------
def FCN(A_prev, W, b, act=ReLU()):
    Z = np.dot(A_prev, W.T) + b.T
    A = act(Z)
    cache = (A_prev, Z)
    return A, cache

def FCN_backward(dA, W, b, cache, act=ReLU()):
    A_prev, Z = cache
    dZ = act.derivative(Z) * dA
    dW = np.dot(dZ.T, A_prev)
    db = np.sum(dZ, axis=0, keepdims=True).T
    dA_prev = np.dot(dZ, W)
    return dA_prev, dW, db

# ---------- SIMPLE CNN CLASS ----------
class SimpleCNN:
    def __init__(self, n_x, n_y, act=ReLU()):
        self.n_x = n_x
        self.n_y = n_y
        self.act = act
        self.init_params()
    def init_params(self):
        self.hparams = {}
        self.params = {}
        f1, p1, s1 = 3, 1, 1
        f2, p2, s2 = 3, 0, 2
        self.hparams['f1'], self.hparams['p1'], self.hparams['s1'] = f1, p1, s1
        self.hparams['f2'], self.hparams['p2'], self.hparams['s2'] = f2, p2, s2
        W1 = np.random.randn(f1, f1) / f1
        b1 = np.zeros((1,))
        W2 = np.random.randn(f2, f2) / f2
        b2 = np.zeros((1,))
        self.n_h = self._calculate_flattened_size()
        W3 = np.random.randn(self.n_y, self.n_h) / np.sqrt(self.n_h)
        b3 = np.zeros((self.n_y, 1))
        self.params['W1'], self.params['b1'] = W1, b1
        self.params['W2'], self.params['b2'] = W2, b2
        self.params['W3'], self.params['b3'] = W3, b3
    def forward(self, X):
        f1, p1, s1 = self.hparams['f1'], self.hparams['p1'], self.hparams['s1']
        f2, p2, s2 = self.hparams['f2'], self.hparams['p2'], self.hparams['s2']
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        self.caches = {}
        A1, cache1 = conv_forward(X, W1, b1, stride=s1, pad=p1, act=self.act)
        A2, cache2 = conv_forward(A1, W2, b2, stride=s2, pad=p2, act=self.act)
        A2_flattened = flatten(A2)
        A3, cache3 = FCN(A2_flattened, W3, b3, act=self.act)
        self.caches['cache1'] = cache1
        self.caches['cache2'] = cache2
        self.caches['cache3'] = cache3
        return A3
    def backward(self, dA3):
        self.grads = {}
        W3, b3 = self.params['W3'], self.params['b3']
        cache3 = self.caches['cache3']
        dA2_flattened, dW3, db3 = FCN_backward(dA3, W3, b3, cache3, act=self.act)
        self.grads['dW3'] = dW3
        self.grads['db3'] = db3
        batch_size = dA2_flattened.shape[0]
        output_size = int(np.sqrt(dA2_flattened.shape[1]))
        dA2 = dA2_flattened.reshape(batch_size, output_size, output_size)
        W2, b2 = self.params['W2'], self.params['b2']
        cache2 = self.caches['cache2']
        f2, p2, s2 = self.hparams['f2'], self.hparams['p2'], self.hparams['s2']
        dA1, dW2, db2 = conv_backward(dA2, W2, b2, cache2, act=self.act, stride=s2, pad=p2)
        self.grads['dW2'] = dW2
        self.grads['db2'] = db2
        W1, b1 = self.params['W1'], self.params['b1']
        cache1 = self.caches['cache1']
        f1, p1, s1 = self.hparams['f1'], self.hparams['p1'], self.hparams['s1']
        dX, dW1, db1 = conv_backward(dA1, W1, b1, cache1, act=self.act, stride=s1, pad=p1)
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1
    def get_params(self):
        return self.params
    def get_grads(self):
        return self.grads
    def get_hparams(self):
        return self.hparams
    def get_caches(self):
        return self.caches
    def _calculate_flattened_size(self):
        f1, p1, s1 = 3, 1, 1
        f2, p2, s2 = 3, 0, 2
        output_size = (self.n_x - f1 + 2 * p1) // s1 + 1
        output_size = (output_size - f2 + 2 * p2) // s2 + 1
        return output_size ** 2

# ---------- GRADIENT DESCENT AND COST ----------
def gradient_descent_step(params, grads, learning_rate):
    for key in params.keys():
        gkey = 'd' + key
        params[key] -= learning_rate * grads[gkey]

def compute_cost(A, Y):
    m = Y.shape[0]
    cost = np.sum((A - Y) ** 2) / (2 * m)
    return cost

# --------- MNIST DATA FOR NUMPY CNN (USE ONLY NUMPY AND TENSORFLOW FOR DATA LOADING) ---------
import tensorflow as tf

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Load and normalize MNIST dataset (only digits 0, 1, 2, for speed)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

classes_to_include = [0, 1, 2]
num_samples_per_class = 100  # Number of samples per class for training

# Filter for the specified classes
train_filter = np.isin(y_train, classes_to_include)
test_filter = np.isin(y_test, classes_to_include)
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

# Randomly select fixed samples per class
def select_random_samples(X, y, classes, num_samples):
    selected_X, selected_y = [], []
    for label in classes:
        indices = np.where(y == label)[0]
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        selected_X.append(X[selected_indices])
        selected_y.append(y[selected_indices])
    return np.concatenate(selected_X), np.concatenate(selected_y)

X_train, y_train = select_random_samples(X_train, y_train, classes_to_include, num_samples_per_class)
X_test, y_test = select_random_samples(X_test, y_test, classes_to_include, num_samples_per_class)

# Prepare input shape: (batch, height, width)
batch_size = X_train.shape[0]
n_x = X_train.shape[1]
n_y = len(classes_to_include)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# One-hot encode labels
y_train_oh = one_hot_encode(y_train, n_y)
y_test_oh = one_hot_encode(y_test, n_y)

# --------- NUMPY CNN TRAINING LOOP ---------
np.random.seed(1)
cnn = SimpleCNN(n_x, n_y, act=ReLU())
learning_rate = 0.01
train_loss = []
test_loss = []

for i in range(10):  # Reduce epochs for speed; increase if needed
    output = cnn.forward(X_test)
    test_loss.append(compute_cost(output, y_test_oh))
    output = cnn.forward(X_train)
    train_loss.append(compute_cost(output, y_train_oh))
    dA3 = (output - y_train_oh) / len(X_train)
    cnn.backward(dA3)
    gradient_descent_step(cnn.get_params(), cnn.get_grads(), learning_rate)
    print(f"Iteration {i}, Train Loss: {train_loss[-1]}, Test Loss: {test_loss[-1]}")

plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('NumPy CNN Loss vs Iteration')
plt.legend()
plt.show()

# SECTION 2: PYTORCH UNET AND IMAGE SEGMENTATION

# ---------- ToIntMask class for mask transformation ----------
class ToIntMask:
    def __call__(self, mask):
        mask = F.pil_to_tensor(mask).long()
        return mask.squeeze(0) - 1

# ---------- IMAGE AND MASK TRANSFORMS ----------
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    ToIntMask()
])

# ---------- DATASET AND DATALOADERS ----------
train_dataset = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation',
                        transform=image_transform, target_transform=mask_transform, download=True)
test_dataset = OxfordIIITPet(root='./data', split='test', target_types='segmentation',
                        transform=image_transform, target_transform=mask_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ---------- CONV BLOCK ----------
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

# ---------- UNET CLASS ----------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._conv_block(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        encodings = []
        for layer in self.encoder:
            x = layer(x)
            encodings.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        encodings = encodings[::-1]
        for encoding, norm_conv, upconv in zip(encodings, self.decoder, self.upconvs):
            x = upconv(x)
            if x.shape != encoding.shape:
                encoding = self.center_crop(encoding, x.shape[2], x.shape[3])
            x = torch.cat([encoding, x], dim=1)
            x = norm_conv(x)
        return self.final_conv(x)
    def _conv_block(self, in_channels, out_channels):
        return conv_block(in_channels, out_channels)
    def center_crop(self, encoding, target_height, target_width):
        _, _, h, w = encoding.size()
        diff_y = (h - target_height) // 2
        diff_x = (w - target_width) // 2
        return encoding[:, :, diff_y:(diff_y + target_height), diff_x:(diff_x + target_width)]

# ---------- TRAIN FUNCTION ----------
def train(model, optimizer, criterion, train_loader, device, num_epochs=10, scheduler=None):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        if scheduler is not None:
            scheduler.step()
    return losses

# ---------- PLOT PREDICTIONS GRID ----------
def plot_predictions_grid(images, masks, predicted_masks, num_samples=9):
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    for idx in range(num_samples):
        image = images[idx].cpu().permute(1, 2, 0).numpy()
        mask = masks[idx].cpu().numpy().squeeze()
        predicted = predicted_masks[idx].cpu().numpy().squeeze()
        axes[idx, 0].imshow(image)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title("Original Image")
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].axis('off')
        axes[idx, 1].set_title("Ground Truth Mask")
        axes[idx, 2].imshow(predicted, cmap='gray')
        axes[idx, 2].axis('off')
        axes[idx, 2].set_title("Predicted Mask")
    plt.tight_layout()
    plt.show()

# --------- UNet TRAINING/TEST LOOP ---------

torch.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = UNet(in_channels=3, out_channels=3).to(device)  # out_channels=3 for 3 classes (pet dataset)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
print("Training UNet...")
train_losses = train(model, optimizer, criterion, train_loader, device, num_epochs=10)

# Predict and plot results for a batch from train
model.eval()
images, masks = next(iter(train_loader))
images = images.to(device)
with torch.no_grad():
    predictions = model(images)
predicted_masks = torch.argmax(predictions, dim=1)
plot_predictions_grid(images, masks, predicted_masks, num_samples=6)

# --------- SAVE AND LOAD MODEL EXAMPLE ---------
epoch = 10
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_losses,
}, f'checkpoint_epoch_{epoch}.pth')
