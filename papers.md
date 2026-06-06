### Phase 1 : L'Inférence Variationnelle (La Base)

L'objectif ici est de maîtriser les fondations des modèles génératifs probabilistes avant d'aborder la diffusion.

1. **Auto-Encoding Variational Bayes** _(Kingma & Welling, 2013)_

- **Pourquoi le lire :** C'est le texte fondateur du VAE. Il formalise l'astuce de reparamétrisation et la dérivation de l'Evidence Lower Bound (ELBO).
- **Le point de blocage mathématique à surmonter :** La démonstration de l'équivalence entre la maximisation de l'ELBO et la minimisation de la divergence de Kullback-Leibler entre le postérior approché et le postérior réel.

**Additional content :**
* https://maitbayev.github.io/posts/auto-encoding-variational-bayes/
* https://www.youtube.com/watch?v=qJeaCHQ1k2w


### Phase 2 : Les Modèles de Diffusion (La Théorie Core)

Une fois l'ELBO compris, on l'applique à une chaîne de Markov pour débruiter des données.

2. **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** _(Sohl-Dickstein et al., 2015)_

- **Pourquoi le lire :** C'est le papier originel qui introduit la diffusion en machine learning. Il est très ancré dans la physique statistique.
- **Le point de blocage mathématique à surmonter :** La formulation de la trajectoire de diffusion complète comme une chaîne de Markov et la dérivation de la borne variationnelle associée.

3. **Denoising Diffusion Probabilistic Models (DDPM)** _(Ho et al., 2020)_

- **Pourquoi le lire :** C'est le papier qui a rendu la diffusion praticable. Les auteurs montrent comment simplifier l'objectif mathématique complexe de Sohl-Dickstein en une simple régression sur le bruit.
- **Le point de blocage mathématique à surmonter :** La démonstration détaillée dans l'annexe (Appendix A) qui montre comment le terme $L_{vlb}$ se factorise et se réduit à $L_{simple}$.

* https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

### Phase 3 : Le Conditionnement et l'Échantillonnage a posteriori (Ton Sujet)

C'est le cœur de ton stage : forcer un modèle à générer quelque chose de précis (ici, un défaut industriel).

4. **Score-Based Generative Modeling through Stochastic Differential Equations** _(Song et al., 2021)_

- **Pourquoi le lire :** Il unifie les modèles de diffusion et les modèles basés sur le score via les équations différentielles stochastiques (SDE). Ce formalisme continu est le plus puissant pour comprendre mathématiquement l'échantillonnage a posteriori.
- **Le point de blocage mathématique à surmonter :** La dynamique de Langevin et le théorème d'Anderson pour inverser une SDE.

5. **Diffusion Models Beat GANs on Image Synthesis** _(Dhariwal & Nichol, 2021)_

- **Pourquoi le lire :** Il introduit le _Classifier Guidance_. C'est l'application directe du théorème de Bayes pour altérer le processus de génération en utilisant le gradient d'un classifieur externe. C'est exactement l'idée du _posterior sampling_ évoquée par ton maître de stage.

### Phase 4 : L'Adaptation Paramétrique (Le Few-Shot)

La dernière brique concerne la modification du modèle pré-entraîné avec très peu de données.

6. **LoRA: Low-Rank Adaptation of Large Language Models** _(Hu et al., 2021)_

- **Pourquoi le lire :** Bien qu'appliqué initialement aux LLMs, la théorie de l'adaptation par factorisation de matrices de faible rang est universelle et c'est celle que tu utiliseras pour adapter le modèle de diffusion aux défauts industriels.
- **Le point de blocage mathématique à surmonter :** Comprendre comment l'initialisation des matrices (une matrice aléatoire, une matrice nulle) garantit que le modèle n'est pas perturbé au début du _fine-tuning_.
