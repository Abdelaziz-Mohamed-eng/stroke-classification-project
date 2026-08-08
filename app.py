import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import colormaps
from tensorflow.keras import Model

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "model.dens121.h5"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Stroke"]  # نفس الترتيب الأبجدي اللي Keras استخدمه وقت التدريب

# لازم القيم دي تكون مطابقة تمامًا لل BOTTOM_FRAC و SIDE_FRAC
# اللي استقريت عليهم في كود التدريب بعد الـ Preview
BOTTOM_FRAC = 0.15
SIDE_FRAC = 0.08

st.set_page_config(page_title="Brain Stroke Classifier", layout="centered")


# ----------------------------
# Load model (cached عشان ميتحملش تاني كل مرة)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


# ----------------------------
# Grad-CAM helpers
# ----------------------------
def crop_borders_pil(image, bottom_frac=BOTTOM_FRAC, side_frac=SIDE_FRAC):
    """
    نفس منطق القص اللي في التدريب بالظبط، بس بـ PIL بدل tf.image
    لإن هنا بنشتغل على صورة واحدة قبل ما تدخل للموديل.
    """
    w, h = image.size
    bottom_crop = int(h * bottom_frac)
    side_crop = int(w * side_frac)

    left = side_crop
    right = w - side_crop
    top = 0
    bottom = h - bottom_crop

    return image.crop((left, top, right, bottom))


def find_last_conv_layer(model):
    """يدور على آخر Conv layer (4D output) جوه الموديل، حتى لو جوه base_model متداخل."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("مفيش Conv layer اتلقى في الموديل")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # آخر layer عندنا هي Dense(1, activation="sigmoid") — بنحسب الـ logits (قبل
    # الـ Sigmoid) يدويًا عشان نتجنب مشكلة الـ Saturated Gradients لما الموديل
    # يبقى واثق جدًا (قريب من 0 أو 1)، لإن ده بيخلي الـ Gradient الحقيقي يقرب من صفر.
    dense_layer = model.layers[-1]

    # موديل فرعي يطلع output آخر conv layer + output الـ layer اللي قبل الـ Dense
    # (يعني الـ GlobalAveragePooling2D output)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.layers[-2].output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, gap_output = grad_model(img_array)
        # بنحسب الـ logit يدويًا: (gap_output . kernel) + bias — من غير Sigmoid
        logits = tf.matmul(gap_output, dense_layer.kernel) + dense_layer.bias
        loss = logits[:, 0]

    # الـ Gradients بتاعة الـ logit بالنسبة لآخر conv layer
    grads = tape.gradient(loss, conv_outputs)

    # متوسط الـ gradients لكل channel (Global Average Pooling على الـ gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize بين 0 و 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(original_image.size)
    )

    jet = colormaps["jet"]
    jet_colors = jet(np.linspace(0, 1, 256))[:, :3]

    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    jet_heatmap_img = Image.fromarray(jet_heatmap)

    return Image.blend(
        original_image.convert("RGB"),
        jet_heatmap_img,
        alpha
    )


# ----------------------------
# UI
# ----------------------------
st.title("Brain Stroke Classification")
st.write("ارفع صورة أشعة (Brain CT/MRI) وهيقولك الموديل الحالة طبيعية ولا فيها Stroke.")

uploaded_file = st.file_uploader("اختار صورة", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة اللي اترفعت")

    if st.button("توقّع النتيجة"):
        with st.spinner("جاري التحليل..."):
            # نفس المعالجة اللي حصلت وقت التدريب: قص الحواف الأول، بعدين Resize
            img_cropped = crop_borders_pil(image)
            img_resized = img_cropped.resize(IMAGE_SIZE)
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]

            threshold = 0.25

            if prediction >= threshold:
                predicted_class = "Stroke"
                confidence = prediction
            else:
                predicted_class = "Normal"
                confidence = 1 - prediction

            # ---- Grad-CAM ----
            last_conv_layer_name = find_last_conv_layer(model)
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            # الـ Overlay بيتحط على نفس الصورة المقصوصة اللي دخلت للموديل فعلًا
            overlayed_image = overlay_heatmap(img_resized, heatmap)

        st.subheader("النتيجة:")
        if predicted_class == "Stroke":
            st.error(f"⚠️ الموديل شايف إن فيه احتمال Stroke — بثقة {confidence:.2%}")
        else:
            st.success(f"✅ الموديل شايف إن الحالة طبيعية (Normal) — بثقة {confidence:.2%}")

        st.caption(f"Raw prediction value: {prediction:.4f}")

        st.subheader("Grad-CAM: الموديل بص فين عشان ياخد القرار")
        st.image(overlayed_image, caption=f"آخر Conv layer اتستخدم: {last_conv_layer_name}")
        st.caption(
            "المناطق الحمرا/الصفرا هي المناطق اللي أثرت في قرار الموديل ناحية Stroke "
            "(الـ Heatmap دلوقتي محسوب من الـ logits قبل الـ Sigmoid، فهو أدق حتى لو "
            "الموديل واثق جدًا من قراره). لو فيه تركيز برّه حدود الجمجمة أو على "
            "الخلفية السودة، ده مؤشر قوي على إن الموديل بيعتمد على Shortcut بدل "
            "الملامح الطبية الحقيقية."
        )

        st.caption("النتيجة دي من موديل تعليمي، ومش بديل عن تشخيص طبي حقيقي.")