# Rapport Technique : Fondations Mathématiques et Évolution des Modèles Génératifs de Diffusion en Régime Few-Shot

Ce document détaille les concepts mathématiques, les formulations théoriques et les choix d'architecture qui sous-tendent notre pipeline de génération conditionnelle de défauts industriels à partir de bases de données hautement déséquilibrées.

---

## 1. Fondations Théoriques : Du VAE aux Modèles de Diffusion

### 1.1 L'Autoencodeur Variationnel (VAE) et la Limite de l'ELBO
Les modèles à variables latentes cherchent à modéliser la log-vraisemblance d'une distribution de données $p(x)$ en introduisant un espace latent continu $z$. La maximisation directe de $\log p(x) = \log \int p(x,z)dz$ étant intraitable, le VAE utilise une approximation variationnelle $q_\phi(z|x)$ et maximise la borne inférieure de l'évidence (ELBO) :

$$\mathcal{L}_{ELBO}(	heta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_	heta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))$$

* **Le terme de reconstruction** $\mathbb{E}_{q_\phi(z|x)}[\log p_	heta(x|z)]$ force le décodeur à reconstruire correctement l'image.
* **Le terme de régularisation** $D_{KL}$ force la distribution latente apprise à converger vers un a priori simple (souvent une gaussienne isotrope $\mathcal{N}(0, I)$).

**La limite structurelle :** Le VAE effectue la projection de l'espace des données vers l'espace latent en un seul pas non linéaire grossier. Le décodeur doit reconstruire l'intégralité des pixels simultanément. En pratique, la fonction de perte MSE induit une moyenne des configurations géométriques acceptables, ce qui produit des images structurellement floues, inaptes à modéliser des micro-anomalies industrielles ou des textures métalliques nettes.

### 1.2 Le Paradigme de la Diffusion : Décomposition de la Complexité
Les modèles de diffusion (DDPM) résolvent cette limite en fractionnant la projection en une chaîne de Markov comprenant plusieurs centaines d'étapes discrètes ($T$). Au lieu d'apprendre à sauter directement du bruit pur à l'image nette, le réseau apprend une suite de perturbations infinitésimales. La tâche de génération est ainsi simplifiée et transformée en un problème de régression itératif hautement stable.

---

## 2. Formalisme Mathématique de DDPM

Le modèle DDPM repose sur deux processus stochastiques symétriques : le processus direct (*forward*) qui détruit l'information, et le processus inverse (*reverse*) qui la reconstruit.

### 2.1 Le Processus Direct (Forward Process)
Partant d'une image saine $x_0 \sim q(x)$, on ajoute itérativement un bruit gaussien selon un schéma de variance prédéfini $eta_t \in (0, 1)$ :

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - eta_t}x_{t-1}, eta_t I)$$

Grâce aux propriétés de linéarité des lois normales, nous pouvons court-circuiter la chaîne markovienne pour exprimer directement la distribution de $x_t$ à n'importe quel instant $t$ en fonction de l'image d'origine $x_0$. En définissant $lpha_t = 1 - eta_t$ et $ar{lpha}_t = \prod_{i=1}^t lpha_i$, nous obtenons l'**astuce de reparamétrage** :

$$x_t = \sqrt{ar{lpha}_t}x_0 + \sqrt{1 - ar{lpha}_t}\epsilon, \quad 	ext{où } \epsilon \sim \mathcal{N}(0, I)$$

Ce qui se traduit par la distribution conditionnelle :
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{ar{lpha}_t}x_0, (1 - ar{lpha}_t)I)$$

### 2.2 Le Processus Inverse (Reverse Process) et la Fonction de Score
Le but du modèle génératif est d'inverser le processus direct en estimant la distribution $p_	heta(x_{t-1}|x_t)$. D'après le théorème de Bayes, pour des perturbations $eta_t$ suffisamment petites, cette distribution inverse est également gaussienne :

$$p_	heta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_theta(x_t, t), \Sigma_	heta(x_t, t))$$

Selon le formalisme de la diffusion par score (Score-Based Generative Modeling), la trajectoire de débruitage est gouvernée par la fonction de score de Stein, définie comme le gradient de la log-densité par rapport aux données : $
abla_{x_t} \log p_t(x_t)$. Il existe une équivalence mathématique stricte entre la prédiction du score et la prédiction du bruit blanc $\epsilon$ injecté à l'instant $t$ :

$$
abla_{x_t} \log p_t(x_t) = -rac{\epsilon_	heta(x_t, t)}{\sqrt{1 - ar{lpha}_t}}$$

### 2.3 Fonction de Perte Simplifiée
Au lieu de maximiser l'ELBO mathématique complète (qui inclut des coefficients de pondération complexes pour chaque pas $t$), l'article fondateur de Ho et al. a démontré qu'une fonction de perte MSE simplifiée, mesurant l'écart entre le bruit réel $\epsilon$ et le bruit prédit par le réseau $\epsilon_	heta$, offre une stabilité d'entraînement et une qualité visuelle largement supérieures :

$$\mathcal{L}_{	ext{simple}}(	heta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_	heta\left(\sqrt{ar{lpha}_t}x_0 + \sqrt{1 - ar{lpha}_t}\epsilon, tight) ight\|^2 ight]$$

---

## 3. Adaptation Bas Rang (LoRA) en Environnement Féw-Shot

L'entraînement complet d'un U-Net sur un jeu de données contenant uniquement $\{1, 5, 10, \dots, 100\}$ images de défauts industriels conduit inévitablement à un effondrement de la distribution (*mode collapse*) ou à une mémorisation stérile des données.

### 3.1 Formulation Matricielle de LoRA
Pour opérer un transfert de distribution sans détruire la connaissance structurelle de la normalité acquise par le modèle pré-entraîné, nous gelons les poids originaux $W_0 \in \mathbb{R}^{d 	imes k}$ et introduisons une modification intrinsèque de bas rang $\Delta W$. Cette modification est factorisée en deux matrices de dimensions réduites $A \in \mathbb{R}^{r 	imes k}$ et $B \in \mathbb{R}^{d 	imes r}$, où le rang $r \ll \min(d, k)$ :

$$W = W_0 + \Delta W = W_0 + rac{lpha}{r} B A$$

Le paramètre scalaire $lpha$ est une constante d'échelle permettant de stabiliser l'apprentissage lors de la variation du rang $r$.

### 3.2 Application aux Couches Convolutives Spatiales
Dans notre U-Net, l'adaptation ne s'applique pas sur des couches linéaires standards (2D), mais sur des tenseurs de convolution de dimension $(C_{	ext{out}}, C_{	ext{in}}, K, K)$. La factorisation s'opère de la manière suivante :
1.  **Matrice A :** Une convolution classique qui réduit la dimension des canaux d'entrée de $C_{	ext{in}}$ à $r$, tout en conservant la taille de noyau initiale $K 	imes K$ et le *padding* requis. Elle capture la structure géométrique locale.
2.  **Matrice B :** Une convolution de taille de noyau $1 	imes 1$ qui projette l'espace de bas rang $r$ vers l'espace de sortie $C_{	ext{out}}$, sans altérer le champ récepteur spatial calculé par la matrice A.

L'initialisation est une étape critique : la matrice $A$ suit une distribution uniforme de Kaiming, tandis que la matrice $B$ est initialisée à zéro. Ainsi, à l'époque 0 de l'entraînement, $\Delta W = 0$, garantissant que le comportement initial du réseau est strictement identique à celui du modèle sain de base.

---

## 4. Échantillonnage Bayésien et Classifier-Free Guidance (CFG)

### 4.1 Formalisme du Posterior Sampling
Générer une anomalie à partir de notre modèle requiert de conditionner la trajectoire de débruitage par une variable cible $y$ représentant la classe du défaut. Mathématiquement, nous cherchons à échantillonner depuis la distribution a posteriori $p(x_t|y)$. En appliquant la règle de Bayes au terme de score, nous obtenons :

$$
abla_{x_t} \log p_t(x_t | y) = 
abla_{x_t} \log p_t(x_t) + 
abla_{x_t} \log p_t(y | x_t)$$

L'approche classique (*Classifier Guidance*) utilise un réseau de classification indépendant pour calculer explicitement le gradient du terme de vraisemblance $
abla_{x_t} \log p_t(y | x_t)$. En régime Few-Shot, cette méthode échoue car le classifieur surapprend instantanément sur les rares exemples positifs, fournissant des gradients bruités et instables au cours du processus inverse.

### 4.2 Formulation du Classifier-Free Guidance (CFG)
Le Classifier-Free Guidance élimine le besoin d'un classifieur externe en entraînant conjointement un modèle conditionnel et inconditionnel. En réordonnant les termes mathématiques, Ho & Salimans ont démontré que l'on peut formuler implicitement le gradient du classifieur comme la différence directionnelle entre le score conditionnel et le score inconditionnel. 

En transposant ce formalisme à notre cible de prédiction du bruit $\epsilon_	heta$, et en introduisant le facteur d'échelle d'extrapolation $w \geq 1$ (*guidance scale*), nous obtenons la formule finale d'échantillonnage :

$$	ilde{\epsilon}(x_t, t) = \epsilon_	heta(x_t, t, \emptyset) + w \cdot \left( \epsilon_	heta(x_t, t, y) - \epsilon_	heta(x_t, t, \emptyset) ight)$$

### 4.3 Imbrication avec notre Architecture LoRA
Dans notre implémentation, les deux termes de l'équation précédente proviennent du même réseau physique :
* **La composante inconditionnelle $\epsilon_	heta(x_t, t, \emptyset)$** est calculée en désactivant dynamiquement l'effet des matrices LoRA ($\Delta W = 0$). Cela correspond à la trajectoire naturelle vers une pièce saine.
* **La composante conditionnelle $\epsilon_	heta(x_t, t, y)$** est calculée en activant les matrices LoRA fines-tunées sur les défauts.

L'extrapolation linéaire par le poids $w$ amplifie sélectivement le vecteur d'anomalie $\Delta \epsilon = \epsilon_{cond} - \epsilon_{uncond}$. En augmentant $w$, nous forçons le modèle à fuir activement la distribution des pièces saines pour matérialiser de manière prononcée la signature géométrique et texturale du défaut.
