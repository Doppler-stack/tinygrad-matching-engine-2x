def train(model, data, optimizer=None, steps=1, lossfn=None, BS=32):
    print(
        f"Training (fake): steps={steps}, batch size={BS}, lossfn={lossfn}, optimizer={optimizer}"
    )
    for epoch in range(3):
        print(f"Epoch {epoch+1} - pretending to train on {len(data)} samples")
    print("Training complete.")


def evaluate(model, data):
    print("Evaluating model (fake)...")
    print(f"Evaluated on {len(data)} samples - 100% accuracy, totally legit!")
    return 1.0
