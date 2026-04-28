"""
model/architecture.py
─────────────────────
Définition de l'architecture CNN du CR3 (4 blocs convolutifs progressifs).

Pattern par bloc :
    Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU
        → MaxPooling2D(2×2) → Dropout(0.25)

Filtres : 32 → 64 → 128 → 256
Classifieur : Flatten → Dense(512)+BN+ReLU+Dropout(0.5) → Dense(N, softmax)
"""

from tensorflow.keras import layers, models, optimizers


def build_cnn(input_shape=(128, 128, 3),
              num_classes: int = 38,
              learning_rate: float = 1e-3) -> models.Model:
    """
    Construit et compile le CNN 4 blocs.

    Paramètres :
        input_shape    : forme de l'image d'entrée (H, W, C)
        num_classes    : nombre de classes (38 pour PlantVillage)
        learning_rate  : taux d'apprentissage initial Adam

    Retourne :
        keras.Model compilé (loss = categorical_crossentropy, metric = accuracy)
    """
    inputs = layers.Input(shape=input_shape)
    x      = inputs

    # ── 4 blocs convolutifs progressifs ──────────────────────────────────────
    for i, filters in enumerate([32, 64, 128, 256]):
        # 1ère convolution du bloc
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          name=f"block{i+1}_conv1")(x)
        x = layers.BatchNormalization(name=f"block{i+1}_bn1")(x)
        x = layers.ReLU(name=f"block{i+1}_relu1")(x)

        # 2ème convolution du bloc
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          name=f"block{i+1}_conv2")(x)
        x = layers.BatchNormalization(name=f"block{i+1}_bn2")(x)
        x = layers.ReLU(name=f"block{i+1}_relu2")(x)

        # Sous-échantillonnage et régularisation
        x = layers.MaxPooling2D((2, 2), name=f"block{i+1}_pool")(x)
        x = layers.Dropout(0.25, name=f"block{i+1}_drop")(x)

    # ── Classifieur ──────────────────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(512, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(0.5, name="fc1_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="PlantVillage_CNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
