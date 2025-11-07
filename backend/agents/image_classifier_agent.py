def _classify_image(self, image_data: np.ndarray, prompt: str) -> dict:
    # Preprocess
    preprocessed = self._preprocess(image_data)
    tensor = torch.from_numpy(preprocessed).unsqueeze(0).float()
    
    # Inference
    with torch.no_grad():
        output = self.model(tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map to labels
    labels = ['BENIGN', 'MALIGNANT', 'NORMAL']
    classification = labels[predicted_class]
    
    return {
        'class': classification,
        'confidence': confidence,
        # ... rest of the logic
    }