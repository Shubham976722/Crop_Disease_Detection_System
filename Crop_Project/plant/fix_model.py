import tensorflow as tf

# LOAD original model
model = tf.keras.models.load_model("models/plant_disease.keras")

# rebuild model (fix compatibility)
model_json = model.to_json()
new_model = tf.keras.models.model_from_json(model_json)
new_model.set_weights(model.get_weights())

# save fixed model
new_model.save("models/final_model.h5")

print("✅ Model fixed successfully")