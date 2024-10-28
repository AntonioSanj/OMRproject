import tensorflow as tf
from keras import layers, Model, applications


# 1. Backbone Network (Feature Extractor)
class Backbone(Model):
    def __init__(self):
        super(Backbone, self).__init__()
        # Use ResNet50 as the backbone, exclude the top layers
        base_model = applications.ResNet50(include_top=False, weights='imagenet')
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block6_out').output)

    def call(self, inputs):
        return self.feature_extractor(inputs)


# 2. Custom RoI Pooling Layer
class RoIPooling(layers.Layer):
    def __init__(self, output_size):
        super(RoIPooling, self).__init__()
        self.output_size = output_size  # e.g., (7, 7)

    def call(self, feature_maps, rois):
        # `rois` is a tensor of shape [num_rois, 5] containing [batch_index, x1, y1, x2, y2]
        # Normalize the RoIs into the range [0, 1] for `crop_and_resize`
        batch_indices = tf.cast(rois[:, 0], dtype=tf.int32)
        boxes = rois[:, 1:] / tf.constant(
            [tf.shape(feature_maps)[2], tf.shape(feature_maps)[1], tf.shape(feature_maps)[2],
             tf.shape(feature_maps)[1]], dtype=tf.float32)
        boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)  # Convert to [y1, x1, y2, x2]

        # Crop and resize to the target size
        cropped_features = tf.image.crop_and_resize(feature_maps, boxes, batch_indices, self.output_size)
        return cropped_features


# 3. Fast R-CNN Head (Classifier and Regressor)
class FastRCNNHead(Model):
    def __init__(self, num_classes):
        super(FastRCNNHead, self).__init__()
        self.num_classes = num_classes

        # Fully connected layers
        self.fc1 = layers.Dense(1024, activation='relu')
        self.fc2 = layers.Dense(1024, activation='relu')

        # Classifier head
        self.cls_score = layers.Dense(num_classes)  # Output classification scores
        # Bounding box regression head (each class has its own 4 values: [dx, dy, dw, dh])
        self.bbox_pred = layers.Dense(num_classes * 4)

    def call(self, x):
        # Forward through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Separate heads for classification and bounding box regression
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


# 4. Fast R-CNN Model
class FastRCNN(Model):
    def __init__(self, num_classes, roi_output_size=(7, 7)):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes

        # Define the backbone network for feature extraction
        self.backbone = Backbone()
        # Define RoI Pooling layer
        self.roi_pooling = RoIPooling(roi_output_size)
        # Define the classification and regression heads
        self.head = FastRCNNHead(num_classes)

    def call(self, images, rois):
        # 1. Extract feature maps using the backbone
        feature_maps = self.backbone(images)

        # 2. Apply RoI Pooling
        pooled_rois = self.roi_pooling(feature_maps,
                                       rois)  # Output shape [num_rois, output_size[0], output_size[1], num_channels]

        # 3. Flatten and pass through the fully connected layers
        pooled_rois = tf.reshape(pooled_rois, [tf.shape(pooled_rois)[0],
                                               -1])  # Flatten each pooled feature to [num_rois, feature_vector_length]

        # 4. Classification and bounding box regression
        cls_score, bbox_pred = self.head(pooled_rois)

        return cls_score, bbox_pred


# 5. Example Usage
if __name__ == "__main__":
    # Create sample data
    batch_size = 2
    num_classes = 21  # Including background
    input_images = tf.random.uniform((batch_size, 224, 224, 3))  # Example batch of 2 images
    rois = tf.constant([[0, 50, 50, 150, 150], [1, 30, 30, 200, 200]],
                       dtype=tf.float32)  # RoIs for each image in the batch

    # Create the Fast R-CNN model
    model = FastRCNN(num_classes)

    # Forward pass
    cls_score, bbox_pred = model(input_images, rois)

    print(f"Classification Scores: {cls_score}")
    print(f"Bounding Box Predictions: {bbox_pred}")
