# Federated Learning
I am currently working on federated learning problems on Riemannian manifolds.
- [Surveys](#Surveys)
- [Riemannian Optimization](#Riemannian_Optimization)
- [Distributed Optimization](#Distributed_Optimization)
- [Applications](#Applications)
- [Links](#Links)

    
<strong> Last Update: 2024/2/5 </strong>


<a name="Surveys" />

## Surveys
- [2024] Federated Learning：Theory and Practice [[Book](https://www.sciencedirect.com/book/9780443190377/federated-learning)] 
- [2024] Trustworthy Distributed AI Systems: Robustness, Privacy, and Governance, ACM Computing Surveys  [[Paper](https://dl.acm.org/doi/abs/10.1145/3645102)]
- [2023] Trusted AI in Multiagent Systems: An Overview of Privacy and Security for Distributed Learning, P IEEE [[Paper](https://ieeexplore.ieee.org/abstract/document/10251703)]
- [2023] Privacy-Preserving Aggregation in Federated Learning: A Survey, IEEE TBDATA [[Paper](https://ieeexplore.ieee.org/abstract/document/9830997)]
- [2023] Communication-Efficient Distributed Learning: An Overview, IEEE JSAC  [[Paper](https://ieeexplore.ieee.org/document/10038471)]
- [2023] 黎曼流形上的无约束优化及其应用, 厦门大学学报  [[Paper](https://kns.cnki.net/kcms2/article/abstract?v=3IEynGI71r9A3umXv4o2sZhoF3qLk5YUcx3c2GNjs5rSheEMx2meSgFCY7NKfgcyc1aC6vd1gdWS-lWub8p_ZhBm4vyHHQSTifJeqxaS65WJOcBqh8dlLRS39EeGIo3MelK3mDKmInZ7IoLN2qy65w==&uniplatform=NZKPT&language=CHS)]
- [2023] An Introduction to Optimization on Smooth Manifolds, Cambridge University Press [[Book](https://www.nicolasboumal.net/book/)] [[Video](https://www.nicolasboumal.net/book/index.html#lectures)]
- [2022] Federated Learning: A Signal Processing Perspective, IEEE SPM  [[Paper](https://ieeexplore.ieee.org/abstract/document/9770266)]
- [2021] A Comprehensive Survey of Privacy-preserving Federated Learning: A Taxonomy, Review, and Future Directions, ACM Computing Surveys [[Paper](https://dl.acm.org/doi/abs/10.1145/3460427)]
- [2021] A Survey on Federated Learning: The Journey From Centralized to Distributed On-Site Learning and Beyond, IEEE JIOT [[Paper](https://ieeexplore.ieee.org/abstract/document/9220780)]
- [2021] Federated Machine Learning: Survey, Multi-Level Classification, Desirable Criteria and Future Directions in Communication and Networking Systems, IEEE COMST [[Paper](https://ieeexplore.ieee.org/document/9352033)]
- [2021] Riemannian Optimization and Its Applications, Springer  [[Book](https://link.springer.com/book/10.1007/978-3-030-62391-3)]
- [2020] 黎曼流形优化及其应用，科学出版社  [[Book](https://item.jd.com/12912482.html)] 
- [2020] A Brief Introduction to Manifold Optimization, Journal of the Operations Research Society of China [[Paper](https://link.springer.com/article/10.1007/s40305-020-00295-9)]
- [2020] First-order and Stochastic Optimization Methods for Machine Learning, Springer [[Book](https://link.springer.com/content/pdf/10.1007/978-3-030-39568-1.pdf)]
- [2020] Secure, Privacy-Preserving and Federated Machine Learning in Medical Imaging, Nature Machine Intelligence [[Paper](https://www.nature.com/articles/s42256-020-0186-1)]
- [2020] Federated Learning: Challenges, Methods, and Future Directions, IEEE SPM [[Paper](https://ieeexplore.ieee.org/abstract/document/9084352)]
- [2020] A Survey on Distributed Machine Learning, ACM Computing Surveys [[Paper](https://dl.acm.org/doi/abs/10.1145/3377454)]
- [2019] A Survey of Distributed Optimization, Annual Reviews in Control [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1367578819300082)]
- [2019] Federated Machine Learning: Concept and Applications, ACM Transactions on Intelligent Systems and Technology  [[Paper](https://dl.acm.org/doi/abs/10.1145/3298981)]
- [2018] A Review of Distributed Algorithms for Principal Component Analysis, P IEEE [[Paper](https://ieeexplore.ieee.org/abstract/document/8425655)]
- [2011] Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers, Foundations and Trends in Machine Learning [[Paper](https://www.nowpublishers.com/article/Details/MAL-016)]
- [2008] Optimization Algorithms on Matrix Manifolds, Princeton University Press [[Book](https://press.princeton.edu/absil)]


<a name="Riemannian_Optimization" />

## Riemannian Optimization


### Feasible Algorithms
- [2021] Multipliers Correction Methods for Optimization Problems Over the Stiefel Manifold, CSIAM Transactions on Applied Mathematics [[Paper](https://www.global-sci.org/intro/article_detail/csiam-am/19448.html)] [[Matlab](https://stmopt.gitee.io/algorithm_description/ProxOrth_code.html)]
- [2018] A New First-Order Algorithmic Framework for Optimization Problems with Orthogonality Constraints, SIAM Journal on Optimization [[Paper](https://epubs.siam.org/doi/abs/10.1137/16M1098759)] [[Matlab](https://www.mathworks.com/matlabcentral/fileexchange/71726-foforth)]
- [2015] A Framework of Constraint Preserving Update Schemes for Optimization on Stiefel Manifold, Mathematical Programming [[Paper](https://link.springer.com/article/10.1007/s10107-014-0816-7)]
- [2013] A Feasible Method for Optimization With Orthogonality Constraints, Mathematical Programming [[Paper](https://link.springer.com/article/10.1007/s10107-012-0584-1)]
- [2012] Projection-like retractions on matrix manifolds, SIAM Journal on Optimization [[Paper](https://epubs.siam.org/doi/abs/10.1137/100802529)]
- [2008] Steepest Descent Algorithms for Optimization Under Unitary Matrix Constraint, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/4436033/)]
- [2007] Trust-region methods on Riemannian manifolds, Foundations of Computational Mathematics [[Paper](https://link.springer.com/article/10.1007/s10208-005-0179-9)]
- [2005] Learning Algorithms Utilizing Quasi-Geodesic Flows on the Stiefel Manifold, Neurocomputing [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231205001128)]
- [2002] Optimization Algorithms Exploiting Unitary Constraints, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/984753)]
- [1998] The Geometry of Algorithms With Orthogonality Constraints, SIAM Journal on Matrix Analysis and Applications [[Paper](https://epubs.siam.org/doi/abs/10.1137/S0895479895290954)]


### Infeasible Algorithms
- [2023] A Riemannian Smoothing Steepest Descent Method for Non-Lipschitz Optimization on Submanifolds, Mathematics of Operations Research  [[Paper](https://pubsonline.informs.org/doi/abs/10.1287/moor.2022.0286)]
- [2023] An Exact Penalty Approach for Optimization with Nonnegative Orthogonality Constraints, Mathematical Programming [[Paper](https://link.springer.com/article/10.1007/s10107-022-01794-8)]
- [2023] A Semismooth Newton Based Augmented Lagrangian Method for Nonsmooth Optimization on Matrix Manifolds, Mathematical Programming [[Paper](https://link.springer.com/article/10.1007/s10107-022-01898-1)] [[Matlab](https://github.com/miskcoo/almssn)]
- [2022] Riemannian Stochastic Proximal Gradient Methods for Nonsmooth Optimization Over the Stiefel Manifold, Journal of Machine Learning Research [[Paper](https://dl.acm.org/doi/abs/10.5555/3586589.3586695)]
- [2022] A Manifold Inexact Augmented Lagrangian Method for Nonsmooth Optimization on Riemannian Submanifolds in Euclidean Space, IMA Journal of Numerical Analysis [[Paper](https://academic.oup.com/imajna/article-abstract/43/3/1653/6590238)] [[Matlab](https://github.com/KKDeng/mialm_code_share)]
- [2022] A Class of Smooth Exact Penalty Function Methods for Optimization Problems With Orthogonality Constraints, Optimization Methods and Software [[Paper](https://www.tandfonline.com/doi/abs/10.1080/10556788.2020.1852236?journalCode=goms20)] [[Matlab](https://stmopt.gitee.io/algorithm_description/PenCF_code.html)]
- [2022] Fast and Accurate Optimization on the Orthogonal Manifold Without Retraction, International Conference on Artificial Intelligence and Statistics [[Paper](https://proceedings.mlr.press/v151/ablin22a)]
- [2021] Exact Penalty Function for L2,1 Norm Minimization Over the Stiefel Manifold, SIAM Journal on Optimization [[Paper](https://epubs.siam.org/doi/abs/10.1137/20M1354313)]  [[Matlab](https://stmopt.gitee.io/algorithm_description/PenCPG_code.html)]
- [2021] Majorization-Minimization on the Stiefel Manifold With Application to Robust Sparse PCA, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/9354027)]
- [2021] Clustering by Orthogonal NMF Model and Non-Convex Penalty Optimization, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/9508841)]
- [2020] Proximal Gradient method for Nonsmooth Optimization Over the Stiefel Manifold, SIAM Journal on Optimization [[Paper](https://epubs.siam.org/doi/abs/10.1137/18M122457X)] [[Matlab](https://github.com/chenshixiang/ManPG)]
- [2019] Parallelizable Algorithms for Optimization Problems With Orthogonality Constraints, SIAM Journal on Scientific Computing [[Paper](https://epubs.siam.org/doi/abs/10.1137/18M1221679)] [[Matlab](https://www.mathworks.com/matlabcentral/fileexchange/71728-pcal)]
- [2019] Structured Quasi-Newton Methods for Optimization With Orthogonality Constraints, SIAM Journal on Scientific Computing [[Paper](https://epubs.siam.org/doi/abs/10.1137/18M121112X)]
- [2018] Adaptive Quadratically Regularized Newton Method for Riemannian Optimization, SIAM Journal on Matrix Analysis and Applications [[Paper](https://epubs.siam.org/doi/abs/10.1137/17M1142478)]
- [2018] A New First-Order Algorithmic Framework for Optimization Problems With Orthogonality Constraints, SIAM Journal on Optimization [[Paper](https://epubs.siam.org/doi/abs/10.1137/16M1098759)] [[Matlab](https://www.mathworks.com/matlabcentral/fileexchange/71726-foforth)]
- [2014] A Splitting Method for Orthogonality Constrained Problems, Journal of Scientific Computing [[Paper](https://link.springer.com/article/10.1007/s10915-013-9740-x)]

  

### Learning Algorithms
- [2023] Learning to Optimize on Riemannian Manifolds, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/9925104)] [[Python](https://github.com/zhigao2017/learningriemannianoptimization)]
- [2021] Orthogonal Deep Neural Networks, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/8877742)] [[Python](https://github.com/Gorilla-Lab-SCUT/OrthDNNs)]
- [2020] Learning to Optimize on SPD Manifolds, CVPR [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Learning_to_Optimize_on_SPD_Manifolds_CVPR_2020_paper.html)] [[Python](https://github.com/zhigao2017/Learning-to-optimize-on-SPD-manifolds)]
- [2018] Can We Gain More from Orthogonality Regularizations in Training Deep Networks?  NIPS [[paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/bf424cb7b0dea050a42b9739eb261a3a-Abstract.html)]  [[Python](https://github.com/VITA-Group/Orthogonality-in-CNNs)]



<a name="Distributed_Optimization" />

## Distributed Optimization

### Distributed Algorithms
- [2022] Distributed Adaptive Newton Methods with Global Superlinear Convergence, Automatica [[Paper](https://www.sciencedirect.com/science/article/pii/S0005109821006865)]
- [2022] Achieving Geometric Convergence for Distributed Optimization with Barzilai-Borwein Step Sizes, Science China Information Sciences  [[Paper](http://scis.scichina.com/en/2022/149204.pdf)]
- [2021] On Distributed Nonconvex Optimization: Projected Subgradient Method for Weakly Convex Problems in Networks, IEEE TAC [[Paper](https://ieeexplore.ieee.org/abstract/document/9345428)]
- [2021] A Penalty Alternating Direction Method of Multipliers for Convex Composite Optimization Over Decentralized Networks, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/9466405)]
- [2021] Decentralized Riemannian gradient descent on the Stiefel manifold, ICML [[Paper](https://proceedings.mlr.press/v139/chen21g.html)] [[Python](https://github.com/chenshixiang/Decentralized_Riemannian_gradient_descent_on_Stiefel_manifold)]
- [2017] Non-Convex Distributed Optimization, IEEE TAC [[Paper](https://ieeexplore.ieee.org/abstract/document/7807315)]
- [2017] Communication-Efficient Learning of Deep Networks from Decentralized Data, ICML [[Paper](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)] [[Python](https://github.com/AshwinRJ/Federated-Learning-PyTorch)]
- [2016] Distributed Compressive Sensing: A Deep Learning Approach, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/7457684)]  [[Matlab](https://github.com/hamidpalangi/Distributed-Compressive-Sensing-A-Deep-Learning-Approach)] 
- [2016] On the Convergence of Decentralized Gradient Descent, SIAM Journal on Optimization [[Paper](https://epubs.siam.org/doi/abs/10.1137/130943170)]
- [2015] EXTRA: An Exact First-Order Algorithm for Decentralized Consensus Optimization, SIAM Journal on Optimization [[Paper](https://epubs.siam.org/doi/abs/10.1137/14096668X)]
- [2010] Distributed Stochastic Subgradient Projection Algorithms for Convex Optimization, Journal of Optimization Theory and Applications [[Paper](https://link.springer.com/article/10.1007/s10957-010-9737-7)]


### Federated Algorithms
- [2023] Decentralized Federated Averaging, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/9850408)]
- [2023] Federated Learning via Inexact ADMM, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/10040221)] [[Matlab](https://github.com/ShenglongZhou/FedADMM)]
- [2023] FedGiA: An Efficient Hybrid Algorithm for Federated Learning, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/10106001)] [[Matlab](https://github.com/ShenglongZhou/FedGiA)]
- [2022] A New Look and Convergence Rate of Federated Multitask Learning With Laplacian Regularization, IEEE TNNLS [[Paper](https://ieeexplore.ieee.org/abstract/document/9975151)] [[Python](https://github.com/CharlieDinh/FedU_FMTL)]
- [2022] Federated Learning Meets Multi-Objective Optimization, IEEE TNSE [[Paper](https://ieeexplore.ieee.org/abstract/document/9762229)]
- [2021] Federated Learning Over Wireless Networks: Convergence Analysis and Resource Allocation, IEEE/ACM TNET [[Paper](https://ieeexplore.ieee.org/abstract/document/9261995)] [[Python](https://github.com/CharlieDinh/FEDL_pytorch)]
- [2020] Personalized Federated Learning with Moreau Envelopes, NIPS [[Paper](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf)] [[Python](https://github.com/CharlieDinh/pFedMe)]


<a name="Applications" />

## Applications

### Subspace Representation
- [2023] Discriminative subspace learning via optimization on Riemannian manifold, Pattern Recognition [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323001504)] [[Matlab](https://github.com/ncclabsustech/MODA-algorithm)]
- [2022] Trace Lasso Regularization for Adaptive Sparse Canonical Correlation Analysis via Manifold Optimization Approach, IEEE TKDE  [[Paper](https://ieeexplore.ieee.org/document/9187560)] [[Matlab](https://github.com/KKDeng/ASCCA)]
- [2021] Robust Low-Rank Matrix Completion via an Alternating Manifold Proximal Gradient Continuation Method, IEEE TSP  [[Paper](https://ieeexplore.ieee.org/abstract/document/9406364)]
- [2020] An Alternating Manifold Proximal Gradient Method for Sparse Principal Component Analysis and Sparse Canonical Correlation Analysis, INFORMS Journal on Optimization [[Paper](https://pubsonline.informs.org/doi/abs/10.1287/ijoo.2019.0032)] [[Matlab](https://github.com/chenshixiang)]
- [2019] Solving Partial Least Squares Regression via Manifold Optimization Approaches, IEEE TNNLS [[Paper](https://ieeexplore.ieee.org/abstract/document/8408735)] [[Matlab](https://github.com/Haoran2014/PLSR_RM)]


### Federated PCA
- [2024] Nonconvex Federated Learning on Compact Smooth Submanifolds With Heterogeneous Data, arXiv [[Paper](https://arxiv.org/abs/2406.08465)]
- [2024] Federated Supervised Principal Component Analysis, IEEE TIFS [[Paper](https://ieeexplore.ieee.org/abstract/document/10292699)]
- [2023] Federated Feature Selection for Horizontal Federated Learning in IoT Networks, JIOT [[Paper](https://ieeexplore.ieee.org/abstract/document/10017376)]
- [2023] Federated PCA on Grassmann Manifold for Anomaly Detection in IoT Networks, IEEE INFOCOM [[Paper](https://ieeexplore.ieee.org/abstract/document/10229026)] [[Python](https://github.com/dual-grp/FedPCA_Abnormal_Detection)]
- [2023] Federated Learning for Sparse Principal Component Analysis, IEEE BigData [[Paper](https://ieeexplore.ieee.org/abstract/document/10386231)]
- [2023] A Communication-Efficient and Privacy-Aware Distributed Algorithm for Sparse PCA, Computational Optimization and Applications [[Paper](https://link.springer.com/article/10.1007/s10589-023-00481-4)] [[C](http://lsec.cc.ac.cn/~liuxin/Solvers/DSSAL1.zip)]
- [2022] Decentralized Optimization Over the Stiefel Manifold by an Approximate Augmented Lagrangian Function, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/9798866)] [[Python](http://lsec.cc.ac.cn/~liuxin/Solvers/DEST.zip)]
- [2022] Seeking Consensus on Subspaces in Federated Principal Component Analysis, arXiv [[Paper](https://arxiv.org/abs/2012.03461)]
- [2022] FAST-PCA: A Fast and Exact Algorithm for Distributed Principal Component Analysis, IEEE TSP [[Paper](https://ieeexplore.ieee.org/abstract/document/10012289)]
- [2021] Fast, Scalable and Geo-distributed PCA for Big Data Analytics, IS [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0306437920301526)] [[Python](https://github.com/tmadnan10/TallnWide)]
- [2021] DeEPCA: Decentralized Exact PCA with Linear Convergence Rate, JMLR [[Paper](https://www.jmlr.org/papers/v22/21-0298.html)]
- [2021] Communication-Efficient Distributed Covariance Sketch, With Application to Distributed PCA, Journal of Machine Learning Research [[Paper](https://dl.acm.org/doi/abs/10.5555/3546258.3546338)]
- [2021] Distributed Principal Component Analysis with Limited Communication, NIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1680e9fa7b4dd5d62ece800239bb53bd-Abstract.html)] [[Matlab](https://github.com/IST-DASLab/QRGD)]
- [2020] Federated Principal Component Analysis, NIPS  [[Paper](https://proceedings.neurips.cc/paper/2020/hash/47a658229eb2368a99f1d032c8848542-Abstract.html)] [[Matlab](https://www.github.com/andylamp/federated_pca)]
- [2020] Communication-Efficient Distributed PCA by Riemannian Optimization, ICML [[Paper](https://proceedings.mlr.press/v119/huang20e.html)] [[Matlab](https://github.com/IST-DASLab/QRGD)]

### Differential Privacy
- [2024] Efficient Sparse Least Absolute Deviation Regression With Differential Privacy, IEEE TIFS [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10379016)]
- [2023] Stochastic privacy-preserving methods for nonconvex sparse learning, Information Sciences  [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025522011161)]
- [2022] Laplacian Smoothing Stochastic ADMMs With Differential Privacy Guarantees, IEEE TIFS [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9762729)]
- [2022] Multiple Strategies Differential Privacy on Sparse Tensor Factorization for Network Traffic Analysis in 5G, IEEE TII [[Paper](https://ieeexplore.ieee.org/abstract/document/9439054)]
- [2022] Gaussian Differential Privacy, Journal of the Royal Statistical Society Series B: Statistical Methodology [[Paper](https://academic.oup.com/jrsssb/article/84/1/3/7056089)]
- [2021] Differentially Private ADMM Algorithms for Machine Learning, IEEE TIFS [[Paper](https://ieeexplore.ieee.org/abstract/document/9540875)]
- [2020] Differential privacy for sparse classification learning, Neurocomputing [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231219312822)]
- [2020] Secure, Privacy-Preserving and Federated Machine Learning in Medical Imaging, Nature Machine Intelligence [[Paper](https://www.nature.com/articles/s42256-020-0186-1)]
- [2019] On Sparse Linear Regression in the Local Differential Privacy Model, ICML [[Paper](http://proceedings.mlr.press/v97/wang19m/wang19m.pdf)] 
- [2018] Minimax-Optimal Privacy-Preserving Sparse PCA in Distributed Systems, ICML [[Paper](http://proceedings.mlr.press/v139/nori21a.html)] 



<a name="Links" />

## Links

### Journals
- IEEE Transactions on Pattern Analysis and Machine Intelligence  [[Link](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34)]
- IEEE Transactions on Signal Processing [[Link](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=78)]
- IEEE Transactions on Automatic Control [[Link](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9)]
- Journal of Machine Learning Research [[Link](https://jmlr.org/)]
- Journal of Scientific Computing  [[Link](https://www.springer.com/journal/10915)]

### Tools
- Awesome Federated Machine Learning  [[Link](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning)] 
- Practicing Federated Learning  [[Link](https://github.com/FederatedAI/Practicing-Federated-Learning)]
- Manopt: A Matlab Toolbox for Optimization on Manifolds  [[Link](https://www.manopt.org/)]
- STOP: A Toolbox for Stiefel Manifold Optimization [[Link](https://stmopt.gitee.io/)]
- Auto-UFSTool: An Automatic MATLAB Toolbox for Unsupervised Feature Selection [[Link](https://github.com/farhadabedinzadeh/AutoUFSTool)]
