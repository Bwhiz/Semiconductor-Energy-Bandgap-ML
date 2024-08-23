# Machine Learning in Semiconductor Materials.

Machine learning (ML) is increasingly being applied in semiconductor materials research to accelerate the discovery, design, and optimization of materials used in electronic devices. Traditionally, semiconductor materials have been studied through experimental methods and theoretical simulations, which can be time-consuming and resource-intensive. ML offers a powerful alternative by enabling predictions of material properties, discovery of new materials, and optimization of manufacturing processes.

### Applications of ML in Semiconductor Materials:

- **Material Property Prediction** : ML models can predict key properties like bandgap, carrier mobility, and thermal conductivity based on the material’s composition and structure. This allows researchers to screen a large number of materials quickly without extensive experiments.

- **Discovery of New Materials**: By analyzing existing datasets, ML algorithms can identify patterns and suggest new semiconductor materials that might have desirable properties, such as higher efficiency or better thermal stability.

- **Process Optimization**: In semiconductor manufacturing, ML is used to optimize processes like deposition, doping, and lithography. ML models can analyze data from manufacturing lines to improve yield, reduce defects, and enhance the overall efficiency of the production process.

- **Defect Detection**: ML algorithms can be employed in real-time defect detection in semiconductor wafers during the manufacturing process, reducing wastage and ensuring higher quality products.

# Why Semiconductors?

Semiconductors are materials that have electrical conductivity between that of a conductor (like copper) and an insulator (like glass). They are the foundation of modern electronics, used in devices like transistors, diodes, and integrated circuits.

Integration of ML into semiconductor research and production represents a significant advancement. By leveraging ML algorithms, the semiconductor industry can reduce development time, improve material performance, and streamline manufacturing processes, ultimately leading to more efficient and cost-effective electronic devices.

# What we Aim to Acheive in this project.

Our Main Objective in this project is to Design a machine learning algorithm that can adequately predict a key property of a material (such as bandgap, carrier mobility, and thermal conductivity ) based on the material’s composition and structure. With main focus on following steps:

- Identifying the main features that can completely describe the element/s.
- Building the dataset based on descriptors.
- Identifying the Model that would accurately predict the dependent variables.
- Rescaling the main features to highlight features with utmost importance.

We present an approach to help scale down the sizes of descriptors to give a better optimization.

### 1. Identifying the main features that can completely describe the element/s.

A major challange and objectives in material science is to generate machine learning (ML) models that can accurately, and rapidly predict a property for a given material by using information derived from the material's structure. However, when the right decriptors are available, predicting material properties such as the energy bandgap would take only few seconds or minimal time using an ML Model, instead of consuming several hours, days or even months to perform. To acheive such an objective, one must find features that can map a material structure in unique material properties. The feature descriptors must be unique to each material and feasible to calculate. An ML model can subsequently be trained to translate descriptors into properties i.e perform the mapping of structure against property. _No matter how sophisticated or "deep" the ML models are, they will fail as long as the descriptors are poorly choosen_

The quality of descriptors is usually by the ability of the descriptors to train predictive ML models. Additionally the following features can be considered as well.

- Meaningfulness of features
- calculation efficiency
- Number of descriptors within the features
- Elemental features
- Geometry-based features
- Electronic structure features
- Ab initio-based features

These decriptors
