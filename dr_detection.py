import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import math

def split_dataset(categories):
    # Load CSV file into a DataFrame
    df = pd.read_csv('images.csv')
    print(df.head())
    id_column = df['id_code'].values  # Converts to a NumPy array

    # Split the IDs into training and testing sets
    train_ids, temp_ids = train_test_split(id_column, test_size=0.2, random_state=42)
    val_ids, test_ids= train_test_split(temp_ids, test_size=0.5, random_state=42)

    print("Train IDs:", len(train_ids))
    print("Val IDs", len(val_ids))
    print("Test IDs:", len(test_ids))

    # Create directories for train/validation/test
    for split in ['train', 'val', 'test']:
        for category in categories:
            path = f'dataset/{split}/{category}'
            if not(os.path.exists(path) and os.path.isdir(path)):
                os.makedirs(path, exist_ok=True)

    return train_ids, val_ids, test_ids

def add_dataset(base_dir, train_ids, val_ids, test_ids, dataset_type):
    for category in categories:
        # List all files in category folder
        images = os.listdir(os.path.join(base_dir, category))
        for img in images:
            img_id = img.split(".png")[0]
            split = None
            if img_id in train_ids:
                split = "train"
            elif img_id in val_ids:
                split = "val"
            elif img_id in test_ids:
                split = "test"

            if dataset_type is not None:
                img_name = f"{img_id}-{dataset_type}.png"
            else:
                img_name = img

            if split is not None:
                shutil.copy(os.path.join(base_dir, category, img), f'dataset/{split}/{category}/{img_name}')



def train_model(train_dir, val_dir):
    # Parameters
    img_size = (224, 224)
    batch_size = 16
    epochs = 10
    fine_tune_epochs = 10

    # Create ImageDataGenerators for training and validation sets
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Create iterators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print(train_generator.samples, train_generator.samples // batch_size)
    print(val_generator.samples, val_generator.samples // batch_size)

    # Load the ResNet50 model with pre-trained ImageNet weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers to prevent updating during initial training
    base_model.trainable = False

    # Add custom layers for DR classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(5, activation='softmax')(x)  # 5 classes for DR severity

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs
    )

    # Fine-Tuning - Unfreeze the base model layers
    base_model.trainable = True

    # Recompile the model with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Fine-tune the model
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=fine_tune_epochs
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
    print(f'Validation accuracy: {val_accuracy:.4f}')

    # Save the trained model
    model.save('dr_classification_model.h5')
    print("Model saved as 'dr_classification_model.h5'")

    return model



def test_model(model_name, test_dir):

    model = load_model(model_name)
    print(model.summary())

    # Create the test data generator with only rescaling
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Create the test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # No shuffling for evaluation
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // 32)
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test loss: {test_loss:.4f}')


if __name__ == "__main__":
    # Assuming images are in subfolders by severity level 
    categories = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
    # train_ids, val_ids, test_ids = split_dataset(categories)

    # Data directories - Update these paths with your dataset location
    normal_dir = "dr_normal/colored_images"
    grayscale_dir = 'dr_grayscale/grayscale_images/grayscale_images'
    gaussian_dir = 'dr_gaussian/gaussian_filtered_images/gaussian_filtered_images'

    # add_dataset(normal_dir, train_ids, val_ids, test_ids, None)
    # add_dataset(grayscale_dir, train_ids, val_ids, test_ids, "grayscale")
    # add_dataset(gaussian_dir, train_ids, val_ids, test_ids, "gaussian")

    # Data directories - Update these paths with your dataset location
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'

    # model = train_model(train_dir, val_dir)

    



    model_file = "dr_classification_model.h5"
    test_model(model_file, test_dir)

