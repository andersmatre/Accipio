
Accipio
========

Accipio will make the famous AI package TensorFlow
easy to use, even with little to no AI experience!

Look how easy it is to use:

    import accipio

    x, y, class_names = accipio.data.load_image_data('images')
    train_images, train_labels, test_image, test_labels = accipio.data.split_data(x, y, split=0.2, shuffle=True, normalize=True)
    model = accipio.ai.image_classification(train_images, train_labels, class_names, epochs=10, batch_size=10)
    predictions = model.predict(test_images)
    accipio.visuals.image_grid(predictions, test_images, test_labels, class_names, cols=5, rows=5)


Installation
------------

Install accipio using pip:

    pip install accipio


Contribute
----------

- Issue Tracker: github.com/andersmatre/accipio/issues
- Source Code: github.com/andersmatre/accipio


License
-------

Accipio is licensed under the Apache License 2.0.