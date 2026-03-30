import tensorflow as tf

# correct path
model = tf.keras.models.load_model("models/plant_disease.keras")

# save in h5 format
model.save("models/model.h5")

print("Model converted successfully ✅")