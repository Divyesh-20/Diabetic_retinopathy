"""
Simple test to check if predictions are inverted
"""
import numpy as np

# If No_DR shows as Proliferative_DR (index 4 instead of 0)
# Then predictions might be reversed

# Theory: The trained model outputs might be in reverse order
# Let's create a fix that inverts the predictions

def invert_predictions(preds):
    """
    Reverse the prediction array.
    This fixes the issue if classes were trained in reverse order.
    """
    return preds[::-1]  # Reverse from [0,1,2,3,4] to [4,3,2,1,0]

# Test
test_preds = np.array([0.1, 0.2, 0.3, 0.25, 0.15])  # Fake prediction favoring class 2
print("Original:", test_preds)
print("Original argmax:", np.argmax(test_preds))

inverted = invert_predictions(test_preds)
print("\nInverted:", inverted)
print("Inverted argmax:", np.argmax(inverted))

# The question: which mapping is correct?
# If user uploaded No_DR (class 0), what does model predict?
print("\nIf user uploaded No_DR and model predicts Proliferative_DR (4),")
print("then predictions are definitely inverted and need fixing.")
