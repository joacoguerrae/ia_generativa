import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


def evaluate(model, criterion, val_loader, device,epoch,save_sample = True,ruta_img = 'samples'):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            # batch puede venir como (x, y) o solo x
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
            else:
                x = batch

            x = x.to(device)                    # [B,3,H,W] en [0,1]

            out = model(x)                      # salida U-Net, típicamente [-1,1] si usás tanh
            out_01 = (out + 1) / 2.0            # la llevamos a [0,1] para VGG

            loss, _, _, _ = criterion(out_01, x)

            val_loss += loss.item()

    if save_sample:
                os.makedirs("samples", exist_ok=True)
                save_image(x[:1],      f"{ruta_img}/content_ep{epoch}.png")
                save_image(out_01[:1], f"{ruta_img}/styled_ep{epoch}.png")

    val_loss /= len(val_loader)
    return val_loss



class EarlyStopping:
    def __init__(self, patience=5):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.val_loss_min = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def print_log(epoch, train_loss, val_loss):
    print(
        f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
    )

def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    do_early_stopping=True,
    patience=5,
    epochs=10,
    log_fn=print_log,
    log_every=1,
    ruta_img = 'samples'
):
    epoch_train_errors = []
    epoch_val_errors = []

    if do_early_stopping:
        early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # batch puede venir como (x, y) o solo x
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch          # ignoramos y (labels)
            else:
                x = batch

            x = x.to(device)          # [B,3,256,256] en [0,1]

            optimizer.zero_grad()

            out = model(x)            # [-1,1] si tu U-Net tiene tanh
            out_01 = (out + 1) / 2.0  # [0,1] para pasar a VGG en la PerceptualLoss

            loss, lc, ls, ltv = criterion(out_01, x)
            #print(f"loss={loss.item():.3f} content={lc.item():.3f} style={ls.item():.3f} tv={ltv.item():.6f}")

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        epoch_train_errors.append(train_loss)

        val_loss = evaluate(model, criterion, val_loader, device,epoch,ruta_img=ruta_img)
        epoch_val_errors.append(val_loss)

        if do_early_stopping:
            early_stopping(val_loss)

        if log_fn is not None and (epoch + 1) % log_every == 0:
            log_fn(epoch, train_loss, val_loss)

        if do_early_stopping and early_stopping.early_stop:
            print(
                f"Detener entrenamiento en la época {epoch}, la mejor pérdida fue {early_stopping.best_score:.5f}"
            )
            break

    return epoch_train_errors, epoch_val_errors



def plot_taining(train_errors, val_errors):
    # Graficar los errores
    plt.figure(figsize=(10, 5))  # Define el tamaño de la figura
    plt.plot(train_errors, label="Train Loss")  # Grafica la pérdida de entrenamiento
    plt.plot(val_errors, label="Validation Loss")  # Grafica la pérdida de validación
    plt.title("Training and Validation Loss")  # Título del gráfico
    plt.xlabel("Epochs")  # Etiqueta del eje X
    plt.ylabel("Loss")  # Etiqueta del eje Y
    plt.legend()  # Añade una leyenda
    plt.grid(True)  # Añade una cuadrícula para facilitar la visualización
    plt.show()  # Muestra el gráfico


def model_calassification_report(model, dataloader, device, nclasses):
    # Evaluación del modelo
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcular precisión (accuracy)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Reporte de clasificación
    report = classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(nclasses)]
    )
    print("Reporte de clasificación:\n", report)


def show_tensor_image(tensor, title=None, vmin=None, vmax=None):
    """
    Muestra una imagen representada como un tensor.

    Args:
        tensor (torch.Tensor): Tensor que representa la imagen. Size puede ser (C, H, W).
        title (str, optional): Título de la imagen. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    # Check if the tensor is a grayscale image
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
    else:  # Assume RGB
        plt.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_tensor_images(tensors, titles=None, figsize=(15, 5), vmin=None, vmax=None):
    """
    Muestra una lista de imágenes representadas como tensores.

    Args:
        tensors (list): Lista de tensores que representan las imágenes. El tamaño de cada tensor puede ser (C, H, W).
        titles (list, optional): Lista de títulos para las imágenes. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    num_images = len(tensors)
    _, axs = plt.subplots(1, num_images, figsize=figsize)
    for i, tensor in enumerate(tensors):
        ax = axs[i]
        # Check if the tensor is a grayscale image
        if tensor.shape[0] == 1:
            ax.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        else:  # Assume RGB
            ax.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        if titles and titles[i]:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()
