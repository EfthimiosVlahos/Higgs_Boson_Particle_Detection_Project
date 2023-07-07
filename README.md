# Higgs Boson Particle Detection Project: Efthimios Vlahos
<img src = "https://raw.githubusercontent.com/sugatagh/Higgs-Boson-Event-Detection/main/Image/atlas_experiment.png" alt = "Higgs into fermions: Evidence of the Higgs boson decaying to fermions" width = "1000">

# Table of contents

- [00. Introduction](#introduction-main)
- [01. Data](#data-overview)
- [02. EDA](#data-eda)
- [03. ANN](#ann-summary)



# Introduction <a name="introduction-main"></a>
## Backstory
**Particle accelerators.** To probe into the basic questions on how matter, space and time work and how they are structured, physicists focus on the simplest interactions (for example, collision of [**subatomic particles**](https://en.wikipedia.org/wiki/Subatomic_particle)) at very high energy. [**Particle accelerators**](https://en.wikipedia.org/wiki/Particle_accelerator) enable physicists to explore the fundamental nature of matter by observing subatomic particles produced by high-energy collisions of [**particle beams**](https://en.wikipedia.org/wiki/Particle_beam). The experimental measurements from these collisions inevitably lack precision, which is where **machine learning** (ML) comes into picture. The research community typically relies on standardized machine learning software packages for the analysis of the data obtained from such experiments and spends a huge amount of effort towards improving statistical power by extracting features of significance, derived from the raw measurements.

**Higgs boson.** The [**Higgs boson**](https://en.wikipedia.org/wiki/Higgs_boson) [**particle**](https://en.wikipedia.org/wiki/Elementary_particle), also called the **God particle** in mainstream media, is the final ingredient of the [**standard model**](https://en.wikipedia.org/wiki/Standard_Model) of [**particle physics**](https://en.wikipedia.org/wiki/Particle_physics), which sets the rules for the subatomic particles and forces. The [**elementary particles**](https://en.wikipedia.org/wiki/Elementary_particle) are supposed to be massless at very high energies, but some of them can acquire mass at low-energies. The mechanism of this acquiring remained an enigma in theoretical physics for a long time. In $1964$, [**Peter Higgs**](https://en.wikipedia.org/wiki/Peter_Higgs) and others proposed a [**mechanism**](https://en.wikipedia.org/wiki/Higgs_mechanism) that theoretically explains the [**origin of mass of elementary particles**](https://en.wikipedia.org/wiki/Mass_generation). The mechanism involves a **field**, commonly known as [**Higgs field**](https://en.wikipedia.org/wiki/Higgs_mechanism#Structure_of_the_Higgs_field), that the paricles can interact with to gain mass. The more a particle interacts with it, the heavier it is. Some particles, like [**photon**](https://en.wikipedia.org/wiki/Photon), do not interact with this field at all and remain massless. The Higgs boson particle is the associated particle of the Higgs field (all fundamental fields have one). It is essentially the physical manifestation of the Higgs field, which gives mass to other particles. The detection of this elusive particle waited almost half a century since its theorization!

**The discovery.** On 4th July 2012, the [**ATLAS**](https://home.cern/science/experiments/atlas) and [**CMS**](https://home.cern/science/experiments/cms) experiments at **CERN**'s **Large Hadron Collider** announced that both of them had observed a new particle in the mass region around 125 GeV. This particle is consistent with the theorized Higgs boson. This experimental confirmation earned [**François Englert**](https://en.wikipedia.org/wiki/Fran%C3%A7ois_Englert) and Peter Higgs [**The Nobel Prize in Physics 2013**](https://www.nobelprize.org/prizes/physics/2013/summary/)
> "for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles, and which recently was confirmed through the discovery of the predicted fundamental particle, by the ATLAS and CMS experiments at CERN's Large Hadron Collider."

**Giving mass to fermions.** There are many different processes through which the Higgs boson can decay and produce other particles. In physics, the possible transformations a particle can undergo as it decays are referred to as [**channels**](https://atlas.cern/glossary/decay-channel). The Higgs boson has been observed first to decay in three distinct decay channels, all of which are [**boson**](https://en.wikipedia.org/wiki/Boson) pairs. To establish that the Higgs field provides the interaction which gives mass to the fundamental [**fermions**](https://en.wikipedia.org/wiki/Fermion) (particles which follow the [**Fermi-Dirac statistics**](https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics), contrary to the bosons which follow the [**Bose-Einstein statistics**](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_statistics)) as well, it has to be demonstrated that the Higgs boson can decay into fermion pairs through direct [**decay**](https://en.wikipedia.org/wiki/Particle_decay) modes. Subsequently, to seek evidence on the decay of Higgs boson into fermion pairs (such as [**tau leptons**](https://simple.wikipedia.org/wiki/Tau_lepton) $(\tau)$ or [**b-quarks**](https://en.wikipedia.org/wiki/Bottom_quark)) and to precisely measure their characteristics became one of the important lines of enquiry. Among the available modes, the most promising is the decay to a pair of tau leptons, which balances a modest branching ratio with manageable backgrounds.

## LHC at Work
**Proton-proton collisions.** In particle physics, an *event* refers to the results just after a [**fundamental interaction**](https://en.wikipedia.org/wiki/Fundamental_interaction) took place between subatomic particles, occurring in a very short time span, at a well-localized region of space. In the LHC, swarms of protons are accelerated on a circular trajectory in both directions, at an extremely high speed. These swarms are made to cross in the **ATLAS** detector, causing hundreds of millions of proton-proton collisions per second. The resulting **events** are detected by sensors, producing a sparse vector of about a hundred thousand dimensions (roughly corresponding to an image or speech signal in classical machine learning applications). The feature construction phase involves extracting type, energy, as well as $3$-D direction of each particle from the raw data. Also, the variable-length list of four-tuples is digested into a fixed-length vector of features containing up to tens of real-valued variables.

**Background events, signal events and selection region.** Some of these variables are first used in a real-time multi-stage cascade classifier (called the trigger) to discard most of the uninteresting events (called **background events**). The selected events (roughly four hundred per second) are then written on disks by a large CPU farm, producing petabytes of data per year. The saved events still, in large majority, represent known processes (these are also *background events*). The background events are mostly produced by the decay of particles which, though exotic in nature, are known beforehand from previous generations of experiments. The goal of the offline analysis is to find a region (called **selection region**) in the feature space that produces significantly excess of events (called **signal events**) compared to what known background processes can explain. Once the region has been fixed, a statistical test is applied to determine the significance of the excess. If the probability that the excess has been produced by background processes falls below a certain limit, it indicates the discovery of a new particle.

**The classification problem.** To optimize the selection region, multivariate classification techniques are routinely utilized. The formal objective function is unique and somewhat different from the classification error or other objectives that are used regularly in machine learning. Nevertheless, finding a *pure* signal region corresponds roughly to separating background events and signal events, which is a standard classification problem. Consequently, established classification methods are useful, as they provide better discovery sensitivity than traditional, manual techniques.

**Weighting and normalization.** The classifier is trained on simulated background events and signal events. Simulators produce weights for each event to correct for the mismatch between the prior probability of the event and the instrumental probability applied by the simulator. The weights are normalized such that in any region, the sum of the weights of events falling in the region gives an unbiased estimate of the expected number of events found there for a fixed integrated luminosity, which corresponds to a fixed data taking time for a given beam intensity. In this case, it corresponds to the data collected by the **ATLAS** experiment in $2012$. Since the probability of a signal event is usually several orders of magnitudes lower than the probability of a background event, the signal samples and the background samples are usually renormalized to produce a balanced classification problem. A real-valued discriminant function is then trained on this reweighted sample to minimize the weighted classification error. The signal region is then defined by cutting the discriminant value at a certain threshold, which is optimized on a held-out set to maximize the sensitivity of the statistical test.

**The broad goal is to improve the procedure that produces the selection region, i.e. the region (not necessarily connected) in the feature space which produces signal events.**


## Enter ML

**Shallow neural network.** Machine learning plays a major role in processing data resulting from experiments at particle colliders. The ML classifiers learn to distinguish between different types of collision events by training on simulated data from sophisticated Monte-Carlo programs. Shallow [**neural networks**](https://en.wikipedia.org/wiki/Neural_network) with single hidden layer are one of the primary techniques used for this analysis and standardized implementations are included in the prevalent multivariate analysis software tools used by physicists. Efforts to increase statistical power tend to focus on developing new features for use with the existing machine learning classifiers. These high-level features are non-linear functions of the low-level measurements, derived using knowledge of the underlying physical processes.

**Deep neural network.** The abundance of labeled simulation training data and the complex underlying structure make this an ideal application for **deep learning**, in particular for large, [**deep neural networks**](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks). Deep neural networks can simplify and improve the analysis of high-energy physics data by automatically learning high-level features from the data. In particular, they increase the statistical power of the analysis even without the help of manually derived high-level features.

# Data <a name="data-overview"></a>
**Source:** **https://www.kaggle.com/competitions/higgs-boson/data**

**The simulator.** The dataset has been built from official ATLAS full-detector simulation. The simulator has two parts. In the first, random proton-proton collisions are simulated based on the knowledge that we have accumulated on particle physics. It reproduces the random microscopic explosions resulting from the proton-proton collisions. In the second part, the resulting particles are tracked through a virtual model of the detector. The process yields simulated events with properties that mimic the statistical properties of the real events with additional information on what has happened during the collision, before particles are measured in the detector.

**Signal sample and background sample.** The signal sample contains events in which Higgs bosons (with a fixed mass of $125$ [**GeV**](https://en.wikipedia.org/wiki/Electronvolt)) were produced. The background sample was generated by other known processes that can produce events with at least one electron or muon and a hadronic tau, mimicking the signal. Only three background processes were retained for the dataset. The first comes from the decay of the $Z$ boson (with a mass of $91.2$ GeV) into two taus. This decay produces events with a topology very similar to that produced by the decay of a Higgs. The second set contains events with a pair of top quarks, which can have a lepton and a hadronic tau among their decay. The third set involves the decay of the $W$ boson, where one electron or muon and a hadronic tau can appear simultaneously only through imperfections of the particle identification procedure.

**Training set and test set.** The training set and the test set respectively contains $250000$ and $550000$ observations. The two sets share $31$ common features between them. Additionally, the training set contains **labels** (**signal** or **background**) and **weights**.


# EDA <a name="data-eda"></a>
**Sneak Peak**
<img width="1140" alt="Screenshot 2023-07-06 at 7 23 07 PM" src="https://github.com/EfthimiosVlahos/Higgs_Boson_Particle_Detection_Project/assets/56899588/f83eb0a2-0341-4c00-b544-934046dccdf5">


**The objective of the project is to classify an event produced in the particle accelerator as background or signal**. As described earlier, a **background event** is explained by the existing theories and previous observations. A **signal event**, however, indicates a process that cannot be described by previous observations and leads to the potential discovery of a new particle.

Exploratory Data Analysis (EDA) plays a pivotal role in this competition, as it enables participants to gain valuable insights into the dataset collected from the Large Hadron Collider (LHC). Here are some of the key steps taken:

**Step 1**: Understanding the Dataset
The initial step in EDA for the Higgs Boson competition involves gaining a deep understanding of the provided dataset. This includes studying the data documentation and familiarizing oneself with the various features, variables, and their corresponding meanings. The dataset typically consists of a vast array of experimental measurements, including particle energies, momenta, and angles, alongside labels indicating the presence or absence of the Higgs Boson particle.

**Step 2**: Data Cleaning and Preprocessing
Before delving into the analysis, it is crucial to ensure that the dataset is free from missing values, outliers, and inconsistencies. Data cleaning involves techniques such as imputation for missing values, outlier detection and handling, and correcting any inconsistencies or errors in the dataset. Additionally, feature engineering techniques can be employed to create new features that may enhance the predictive power of the models.

**Step 3**: Statistical Summaries and Visualizations
The next phase of EDA focuses on generating statistical summaries and visualizations to gain deeper insights into the dataset. Descriptive statistics, such as mean, median, standard deviation, and correlation coefficients, provide a quantitative understanding of the data. Visualizations, including histograms, scatter plots, box plots, and heatmaps, offer a graphical representation of the data's distribution, relationships, and potential patterns. These analyses assist in identifying any inherent biases, trends, or anomalies present in the dataset.

**Step 4**: Feature Selection and Dimensionality Reduction
Feature selection is a crucial aspect of EDA, aiming to identify the most relevant variables that contribute significantly to the prediction of the Higgs Boson particle. Techniques such as correlation analysis, mutual information, or feature importance from machine learning models can aid in selecting the most informative features. 

**Step 5**: Exploring Relationships and Dependencies
EDA involves investigating the relationships and dependencies between variables in the dataset. This can be accomplished through further visualizations, such as scatter plots, pair plots, or correlation matrices. It helps in identifying strong correlations or dependencies between variables, which can guide the feature engineering process or influence the choice of predictive models.

**Step 6**: Data Sampling and Model Validation
To ensure the generalizability of the models developed, it is essential to split the dataset into training, validation, and test sets. EDA can guide this process by helping identify any class imbalances or biases in the dataset, allowing for appropriate sampling techniques like stratified sampling. Additionally, exploring the distribution of the target variable across different subsets of data aids in understanding potential biases and challenges in model development and validation.

**Step 7**: Outlier Detection and Anomaly Analysis
EDA is instrumental in detecting outliers or anomalies in the dataset that might adversely affect model performance. Techniques such as statistical methods (e.g., z-score, interquartile range) or machine learning algorithms (e.g., Isolation Forest, Local Outlier Factor) can be utilized to identify and handle outliers effectively. Understanding the nature and characteristics of these outliers can provide valuable insights into the data generation process.

**Conclusion**:
Exploratory Data Analysis (EDA) serves as a crucial step in the Higgs Boson competition, offering participants a deeper understanding of the dataset collected from the Large Hadron Collider (LHC). By comprehensively exploring the dataset, cleaning and preprocessing the data, generating statistical summaries, visualizing relationships, selecting relevant features, and validating models, data scientists can unravel the intricate secrets of the Higgs Boson particle. EDA empowers researchers to make informed decisions, enhance the predictive power of models, and contribute to cutting-edge research in particle physics.


# ANN <a name="ann-summary"></a>

The developed artificial neural network (ANN) model using Keras functional APIs for the Higgs Boson competition achieved accurate predictions, as evaluated through rigorous evaluation metrics. By leveraging the power of ANNs, the model effectively captured complex patterns and relationships present in the data, allowing for robust predictions. The Keras functional APIs provided a flexible and intuitive framework to construct the ANN architecture, enabling the integration of various layers and activation functions.

To improve the model's performance, particular attention was given to ensuring a balanced class distribution within the dataset. By equalizing the representation of both signal and background noise classes, the model was able to mitigate bias and improve its ability to generalize to different class instances. This balanced class distribution prevented the model from favoring one class over the other, ensuring fairness and reliability in predictions.

The ANN exhibited exceptional performance on unseen data, which was assessed through a comprehensive test set evaluation. With a precision rate of 92%, the model demonstrated its capability to generalize well to new, unseen instances and make accurate predictions. This high precision value indicated that the model had a low false positive rate, implying that it made few incorrect predictions of background noise as signal or vice versa. The ability to perform well on unseen data showcases the model's robustness and reliability in real-world scenarios.

In conclusion, the developed ANN model using Keras functional APIs for the Higgs Boson competition achieved accurate predictions by effectively capturing complex patterns and relationships in the data. The model's performance was further enhanced by ensuring a balanced class distribution, mitigating bias and promoting generalization. With an exceptional precision rate of 92% on the test set, the model demonstrated its ability to make accurate predictions on unseen data, establishing its reliability and practical applicability.










