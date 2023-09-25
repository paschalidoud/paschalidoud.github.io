#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from collections import namedtuple
import sys


def _print(args, text, **kwargs):
    if not args.quiet:
        print(text, **kwargs)


class Author(namedtuple("Author", ["name", "url"])):
    @property
    def is_me(self):
        return self.name == "Despoina Paschalidou"


class Paper(namedtuple("Paper", [
        "title",
        "url",
        "image",
        "authors",
        "conference",
        "year",
        "special",
        "links"
    ])):
    pass


class Conference(namedtuple("Conference", ["name"])):
    pass


class Link(namedtuple("Link", ["name", "url", "html", "text"])):
    pass


def author_list(authors, *names):
    return [authors[n] for n in names]


authors = {
    "despi": Author("Despoina Paschalidou", ""),
    "osman": Author("Ali Osman Ulusoy", "https://scholar.google.de/citations?user=fkqdDEEAAAAJ&hl=en"),
    "andreas": Author("Andreas Geiger", "http://www.cvlibs.net/"),
    "sanja": Author("Sanja Fidler", "https://www.cs.utoronto.ca/~fidler/"),
    "leo": Author("Leonidas Guibas", "https://geometry.stanford.edu/member/guibas/"),
    "minhyuk": Author("Minhyuk Sung", "https://mhsung.github.io/"),
    "alec": Author("Alec Jacobson", "https://www.cs.toronto.edu/~jacobson/"),
    "konstantinos": Author("Konstantinos Tertikas", "https://ktertikas.github.io/"),
    "zhen": Author("Zhen Wang", "https://zhenwangwz.github.io/"),
    "shijie": Author("Shijie Zhou", "https://www.linkedin.com/in/shijie-zhou-ucla"),
    "sherwin": Author("Sherwin Bahmani", "https://sherwinbahmani.github.io/"),
    "jj": Author("Jeong Joon Park", ""),
    "suya": Author("Suya You", ""),
    "hao": Author("Hao Tang", ""),
    "xingguang": Author("Xingguang Yan", "http://yanxg.art/"),
    "andrea": Author("Andrea Tagliasacchi", "https://taiya.github.io/"),
    "achuta": Author("Achuta Kadambi", "https://www.ee.ucla.edu/achuta-kadambi/"),
    "mika": Author("Mikaela Angelina Uy", "https://mikacuy.github.io/"),
    "yannis": Author("Yannis Avrithis", "https://avrithis.net/"),
    "emiris": Author("Ioannis Emiris", "https://cgi.di.uoa.gr/~emiris/index-eng.html"),
    "gordon": Author("Gordon Wetzstein", "https://stanford.edu/~gordonwz/"),
    "radu": Author("Radu Timofte", "http://people.ee.ethz.ch/~timofter/"),
    "boxiao": Author("Boxiao Pan", "https://cs.stanford.edu/~bxpan/"),
    "will": Author("Bokui Shen", "https://cs.stanford.edu/people/bshen88/"),
    "davis": Author("Davis Rempe", "https://davrempe.github.io/"),
    "kaichun": Author("Kaichun Mo", "https://cs.stanford.edu/~kaichun/"),
    "yanchao": Author("Yanchao Yang", "https://yanchaoyang.github.io/"),
    "amlan": Author("Amlan Kar", "https://amlankar.github.io/"),
    "masha": Author("Maria Shugrina", "http://shumash.com/"),
    "karsten": Author("Karsten Kreis", "https://scholar.google.de/citations?user=rFd-DiAAAAAJ&hl=de"),
    "aseem": Author("Aseem Behl", "http://aseembehl.github.io/"),
    "simon": Author("Simon Donné", "https://donnessime.github.io/"),
    "caro": Author("Carolin Schmitt", "https://avg.is.tuebingen.mpg.de/person/cschmitt"),
    "kunho": Author("Kunho Kim", "https://soulmates2.github.io/CV/"),
    "luc": Author("Luc van Gool", "https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html"),
    "angelos": Author("Angelos Katharopoulos", "https://angeloskath.github.io/"),
    "diou": Author("Christos Diou", "https://mug.ee.auth.gr/people/christos-diou/"),
    "delo": Author("Anastasios Delopoulos", "https://mug.ee.auth.gr/people/anastasios-delopoulos/")
}
conferences = {
    "neurips": Conference("Advances in Neural Information Processing Systems (NeurIPS)"),
    "cvpr": Conference("Computer Vision and Pattern Recognition (CVPR)"),
    "iccv": Conference("International Conference on Computer Vision (ICCV)"),
    "eusipco": Conference("European Signal Processing Conference (EUSIPCO)"),
    "acmmm": Conference("ACM Multimedia Conference (ACMM)"),
    "tmlr": Conference("Transactions on Machine Learning Research (TMLR)"),
    "pg": Conference("Pacific Graphics"),
    "arxiv": Conference("arXiv")
}
publications = [
    Paper(
        "OptCtrlPoints: Finding the Optimal Control Points for Biharmonic 3D Shape Deformation",
        "https://soulmates2.github.io/publications/OptCtrlPoints/",
        "teasers/optctrl_teaser.png",
        author_list(authors, "kunho", "mika", "despi", "alec", "leo", "minhyuk"),
        conferences["pg"],
        2023,
        None,
        [   Link("Abstract", None, "We propose OptCtrlPoints, a data-driven framework designed to identify the optimal sparse set of control points for reproducing target shapes using biharmonic 3D shape deformation. Control-point-based 3D deformation methods are widely utilized for interactive shape editing, and their usability is enhanced when the control points are sparse yet strategically distributed across the shape. With this objective in mind, we introduce a data-driven approach that can determine the most suitable set of control points, assuming that we have a given set of possible shape variations. The challenges associated with this task primarily stem from the computationally demanding nature of the problem. Two main factors contribute to this complexity: solving a large linear system for the biharmonic weight computation and addressing the combinatorial problem of finding the optimal subset of mesh vertices. To overcome these challenges, we propose a reformulation of the biharmonic computation that reduces the matrix size, making it dependent on the number of control points rather than the number of vertices. Additionally, we present an efficient search algorithm that significantly reduces the time complexity while still delivering a nearly optimal solution. Experiments on SMPL, SMAL, and DeformingThings4D datasets demonstrate the efficacy of our method. Our control points achieve better template-to-target fit than FPS, random search, and neural-network-based prediction. We also highlight the significant reduction in computation time from days to approximately 3 minutes.", None),
            Link("Project page", "https://soulmates2.github.io/publications/OptCtrlPoints/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2309.12899.pdf", None, None),
            Link("Bibtex", None, None, """@Article{Kim2023PG,
      title={OptCtrlPoints: Finding the Optimal Control Points for Biharmonic 3D Shape Deformation}, 
      author={Kunho Kim and Mikaela Angelina Uy and Despoina Paschalidou and Alec Jacobson and Leonidas J. Guibas and Minhyuk Sung},
      journal={arXiv preprint arXiv:2309.12899},
      year={2023},
    }
""")
        ]
    ),

    Paper(
        "CC3D: Layout-Conditioned Generation of Compositional 3D Scenes",
        "https://sherwinbahmani.github.io/cc3d/",
        "teasers/cc3d_teaser_2.png",
        author_list(authors, "sherwin", "jj", "despi", "xingguang", "gordon", "leo", "andrea"),
        conferences["iccv"],
        2023,
        None,
        [   Link("Abstract", None, "In this work, we introduce CC3D, a conditional generative model that synthesizes complex 3D scenes conditioned on 2D semantic scene layouts, trained using single-view images. Different from most existing 3D GANs that limit their applicability to aligned single objects, we focus on generating complex scenes with multiple objects, by modeling the compositional nature of 3D scenes. By devising a 2D layoutbased approach for 3D synthesis and implementing a new 3D field representation with a stronger geometric inductive bias, we have created a 3D GAN that is both efficient and of high quality, while allowing for a more controllable generation process. Our evaluations on synthetic 3D-FRONT and real-world KITTI-360 datasets demonstrate that our model generates scenes of improved visual and geometric quality in comparison to previous works.", None),
            Link("Project page", "https://sherwinbahmani.github.io/cc3d/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2303.12074.pdf", None, None),
            Link("Poster", "data/Bahmani2023ICCV_poster.pdf", None, None),
            Link("Code", "https://github.com/sherwinbahmani/cc3d", None, None),
            Link("Bibtex", None, None, """@InProceedings{Bahmani2023ICCV
  author = {Bahmani, Sherwin and Park, Jeong Joon and Paschalidou, Despoina and Yan, Xingguang and Wetzstein, Gordon and Guibas, Leonidas and Tagliasacchi, Andrea},
  title = {CC3D: Layout-Conditioned Generation of Compositional 3D Scenes},
  booktitle = {International Conference on Computer Vision (ICCV)}},
  year = {2023}
}""")
        ]
    ),

    Paper(
        "COPILOT: Human Collision Prediction and Localization from Multi-view Egocentric Videos",
        "https://sites.google.com/stanford.edu/copilot",
        "teasers/copilot.jpg",
        author_list(authors, "boxiao", "will", "davis", "despi", "kaichun", "yanchao", "leo"),
        conferences["iccv"],
        2023,
        None,
        [   Link("Abstract", None, "The ability to forecast human-environment collisions from egocentric observations is vital to enable collision avoidance in applications such as VR, AR, and wearable assistive robotics. In this work, we introduce the challenging problem of predicting collisions in diverse environments from multi-view egocentric videos captured from body-mounted cameras. Solving this problem requires a generalizable perception system that can classify which human body joints will collide and estimate a collision region heatmap to localize collisions in the environment. To achieve this, we propose a transformer-based model called COPILOT to perform collision prediction and localization simultaneously, which accumulates information across multi-view inputs through a novel 4D space-time-viewpoint attention mechanism. To train our model and enable future research on this task, we develop a synthetic data generation framework that produces egocentric videos of virtual humans moving and colliding within diverse 3D environments. This framework is then used to establish a large-scale dataset consisting of 8.6M egocentric RGBD frames. Extensive experiments show that COPILOT generalizes to unseen synthetic as well as real-world scenes. We further demonstrate COPILOT outputs are useful for downstream collision avoidance through simple closed-loop control." , None),
            Link("Project page", "https://sites.google.com/stanford.edu/copilot", None, None),
            Link("Paper", "https://arxiv.org/pdf/2210.01781.pdf", None, None),
            Link("Video", "https://www.youtube.com/watch?v=lxRTPeac8Oo", None, None),
            Link("Bibtex", None, None, """@InProceedings{Pan2023ICCV,
  author = {Boxiao Pan, Bokui Shen, Davis Rempe, Despoina Paschalidou, Kaichun Mo, Yanchao Yang, Leonidas J Guibas},
  title = {COPILOT: Human Collision Prediction and Localization from Multi-view Egocentric Videos},
  booktitle = {International Conference on Computer Vision (ICCV)}},
  year = {2023}
}""")
        ]
    ),

    Paper(
        "3D-Aware Video Generation",
        "https://sherwinbahmani.github.io/3dvidgen/",
        "teasers/3dvidgen.png",
        author_list(authors, "sherwin", "jj", "despi", "hao", "gordon", "leo", "luc", "radu"),
        conferences["tmlr"],
        2023,
        None,
        [   Link("Abstract", None, "Generative models have emerged as an essential building block for many image synthesis and editing tasks. Recent advances in this field have also enabled high-quality 3D or video content to be generated that exhibits either multi-view or temporal consistency. With our work, we explore 4D generative adversarial networks (GANs) that learn unconditional generation of 3D-aware videos. By combining neural implicit representations with time-aware discriminator, we develop a GAN framework that synthesizes 3D video supervised only with monocular videos. We show that our method learns a rich embedding of decomposable 3D structures and motions that enables new visual effects of spatio-temporal renderings while producing imagery with quality comparable to that of existing 3D or video GANs.", None),
            Link("Project page", "https://sherwinbahmani.github.io/3dvidgen/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2206.14797.pdf", None, None),
            Link("Code", "https://github.com/sherwinbahmani/3dvideogeneration/", None, None),
            Link("Bibtex", None, None, """@InProceedings{Bahmani2022TMLR,
  author = {Bahmani, Sherwin and Park, Jeong Joon and Paschalidou, Despoina and Tang, Hao and Wetzstein, Gordon and Guibas, Leonidas and Van Gool, Luc and Timofte, Radu},
  title = {3D-Aware Video Generation},
  booktitle = {arXiv preprint},
  year = {2022}
}""")
        ]
    ),
    
    Paper(
        "PartNeRF: Generating Part-Aware Editable 3D Shapes without 3D Supervision",
        "https://ktertikas.github.io/part_nerf",
        "teasers/partnerf_2.png",
        author_list(authors, "konstantinos", "despi", "boxiao", "jj", "mika", "emiris", "yannis", "leo"),
        conferences["cvpr"],
        2023,
        None,
        [   Link("Abstract", None, "Impressive progress in generative models and implicit representations gave rise to methods that can generate 3D shapes of high quality. However, being able to locally control and edit shapes is another essential property that can unlock several content creation applications. Local control can be achieved with part-aware models, but existing methods require 3D supervision and cannot produce textures. In this work, we devise PartNeRF, a novel part-aware generative model for editable 3D shape synthesis that does not require any explicit 3D supervision. Our model generates objects as a set of locally defined NeRFs, augmented with an affine transformation. This enables several editing operations such as applying transformations on parts, mixing parts from different objects etc. To ensure distinct, manipulable parts we enforce a hard assignment of rays to parts that makes sure that the color of each ray is only determined by a single NeRF. As a result, altering one part does not affect the appearance of the others. Evaluations on various ShapeNet categories demonstrate the ability of our model to generate editable 3D objects of improved fidelity, compared to previous part-based generative approaches that require 3D supervision or models relying on NeRFs.", None),
            Link("Project page", "https://ktertikas.github.io/part_nerf", None, None),
            Link("Paper", "https://arxiv.org/pdf/2303.09554.pdf", None, None),
            Link("Poster", "data/Tertikas2023CVPR_poster.pdf", None, None),
            Link("Slides", "slides/Tertikas2023CVPR_slides.pdf", None, None),
            Link("Code", "https://github.com/ktertikas/part_nerf", None, None),
            Link("Bibtex", None, None, """@InProceedings{Tertikas2023CVPR,
  author    = {Konstantinos Tertikas and Despoina Paschalidou and Boxiao Pan and Jeong Joon Park and Mikaela Angelina Uy and Ioannis Emiris and Yannis Avrithis and Leonidas Guibas},
  title     = {PartNeRF: Generating Part-Aware Editable 3D Shapes without 3D Supervision},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}""")
        ]
    ),
    Paper(
        "ALTO: Alternating Latent Topologies for Implicit 3D Reconstruction",
        "https://visual.ee.ucla.edu/alto.htm/",
        "teasers/alto_teaser.png",
        author_list(authors, "zhen", "shijie", "jj", "despi", "suya", "gordon", "leo", "achuta"),
        conferences["cvpr"],
        2023,
        None,
        [   Link("Abstract", None, "This work introduces alternating latent topologies (ALTO) for high-fidelity reconstruction of implicit 3D surfaces from noisy point clouds. Previous work identifies that the spatial arrangement of latent encodings is important to recover detail. One school of thought is to encode a latent vector for each point (point latents). Another school of thought is to project point latents into a grid (grid latents) which could be a voxel grid or triplane grid. Each school of thought has tradeoffs. Grid latents are coarse and lose high-frequency detail. In contrast, point latents preserve detail. However, point latents are more difficult to decode into a surface, and quality and runtime suffer. In this paper, we propose ALTO to sequentially alternate between geometric representations, before converging to an easy-to-decode latent. We find that this preserves spatial expressiveness and makes decoding lightweight. We validate ALTO on implicit 3D recovery and observe not only a performance improvement over the state-of-the-art, but a runtime improvement of 3-10×.", None),
            Link("Project page", "https://visual.ee.ucla.edu/alto.htm/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2212.04096.pdf", None, None),
            Link("Poster", "data/Zhen2023CVPR_poster.pdf", None, None),
            Link("Slides", "slides/presentation_alto.pdf", None, None),
            Link("Code", "https://github.com/wzhen1/ALTO", None, None),
            Link("Video", "https://www.youtube.com/watch?v=EsnE4dEIArY", None, None),
            Link("Bibtex", None, None, """@InProceedings{Zhen2023CVPR,
    title = {ALTO: Alternating Latent Topologies for Implicit 3D Reconstruction},
    author = {Wang, Zhen and Zhou, Shijie and Park, Jeong Joon and Paschalidou, Despoina and You, Suya and Wetzstein, Gordon and Guibas, Leonidas and Kadambi, Achuta},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023}
}""")
        ]
    ),

    Paper(
        "ATISS: Autoregressive Transformers for Indoor Scene Synthesis",
        "https://nv-tlabs.github.io/ATISS/#",
        "teasers/atiss.png",
        author_list(authors, "despi", "amlan", "masha", "karsten", "andreas", "sanja"),
        conferences["neurips"],
        2021,
        None,
        [   Link("Abstract", None, "The ability to synthesize realistic and diverse indoor furniture layouts automatically or based on partial input, unlocks many applications, from better interactive 3D tools to data synthesis for training and simulation. In this paper, we present ATISS, a novel autoregressive transformer architecture for creating diverse and plausible synthetic indoor environments, given only the room type and its floor plan. In contrast to prior work, which poses scene synthesis as sequence generation, our model generates rooms as unordered sets of objects. We argue that this formulation is more natural, as it makes ATISS generally useful beyond fully automatic room layout synthesis. For example, the same trained model can be used in interactive applications for general scene completion, partial room re-arrangement with any objects specified by the user, as well as object suggestions for any partial room. To enable this, our model leverages the permutation equivariance of the transformer when conditioning on the partial scene, and is trained to be permutation-invariant across object orderings. Our model is trained end-to-end as an autoregressive generative model using only labeled 3D bounding boxes as supervision. Evaluations on four room types in the 3D-FRONT dataset demonstrate that our model consistently generates plausible room layouts that are more realistic than existing methods. In addition, it has fewer parameters, is simpler to implement and train and runs up to 8x faster than existing methods.", None),
            Link("Project page", "https://nv-tlabs.github.io/ATISS/#", None, None),
            Link("Paper", "https://arxiv.org/pdf/2110.03675.pdf", None, None),
            Link("Poster", "data/Paschalidou2021NEURIPS_poster.pdf", None, None),
            Link("Slides", "data/Paschalidou2021NEURIPS_slides.pdf", None, None),
            Link("Code", "https://github.com/nv-tlabs/atiss", None, None),
            Link("Video", "https://www.youtube.com/watch?v=VNY0BFMi2j4", None, None),
            Link("Bibtex", None, None, """@InProceedings{Paschalidou2021NEURIPS,
  author = {Despoina Paschalidou and Amlan Kar and Maria Shugrina and Karsten Kreis and Andreas Geiger and Sanja Fidler},
  title = {ATISS: Autoregressive Transformers for Indoor Scene Synthesis},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}""")
        ]
    ),
    Paper(
        "Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks",
        "https://paschalidoud.github.io/neural_parts",
        "teasers/neural_parts.png",
        author_list(authors, "despi", "angelos", "andreas", "sanja"),
        conferences["cvpr"],
        2021,
        None,
        [   Link("Abstract", None, "Impressive progress in 3D shape extraction led to representations that can capture object geometries with high fidelity. In parallel, primitive-based methods seek to represent objects as semantically consistent part arrangements. However, due to the simplicity of existing primitive representations, these methods fail to accurately reconstruct 3D shapes using a small number of primitives/parts. We address the trade-off between reconstruction quality and number of parts with Neural Parts, a novel 3D primitive representation that defines primitives using an Invertible Neural Network (INN) which implements homeomorphic mappings between a sphere and the target object. The INN allows us to compute the inverse mapping of the homeomorphism, which in turn, enables the efficient computation of both the implicit surface function of a primitive and its mesh, without any additional post-processing. Our model learns to parse 3D objects into semantically consistent part arrangements without any part-level supervision. Evaluations on ShapeNet, D-FAUST and FreiHAND demonstrate that our primitives can capture complex geometries and thus simultaneously achieve geometrically accurate as well as interpretable reconstructions using an order of magnitude fewer primitives than state-of-the-art shape abstraction methods.", None),
            Link("Project page", "https://paschalidoud.github.io/neural_parts", None, None),
            Link("Paper", "https://arxiv.org/pdf/2103.10429.pdf", None, None),
            Link("Poster", "data/Paschalidou2021CVPR_poster.pdf", None, None),
            Link("Code", "https://github.com/paschalidoud/neural_parts", None, None),
            Link("Blog", "https://autonomousvision.github.io/neural-parts/", None, None),
            Link("Slides", "http://www.cvlibs.net/publications/Paschalidou2021CVPR_slides.pdf", None, None),
            Link("Video", "https://www.youtube.com/watch?v=6WK3B0IZJsw", None, None),
            Link("Podcast", "https://www.itzikbs.com/neural-parts-learning-expressive-3d-shape-abstractions-with-invertible-neural-networks", None, None),
            Link("Bibtex", None, None, """@InProceedings{Paschalidou2021CVPR,
    title = {Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks},
    author = {Paschalidou, Despoina and Katharopoulos, Angelos and Geiger, Andreas and Fidler, Sanja},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    month = jun,
    year = {2021}
}""")
        ]
    ),
    Paper(
        "Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image",
        "http:superquadrics.com/hierarchical_primitives",
        "teasers/hierarchical_primitives.png",
        author_list(authors, "despi", "luc", "andreas"),
        conferences["cvpr"],
        2020,
        None,
        [
            Link("Abstract", None, "Humans perceive the 3D world as a set of distinct objects that are characterized by various low-level (geometry, reflectance) and high-level (connectivity, adjacency, symmetry) properties. Recent methods based on convolutional neural networks (CNNs) demonstrated impressive progress in 3D reconstruction, even when using a single 2D image as input. However, the majority of these methods focuses on recovering the local 3D geometry of an object without considering its part-based decomposition or relations between parts. We address this challenging problem by proposing a novel formulation that allows to jointly recover the geometry of a 3D object as a set of primitives as well as their latent hierarchical structure without part-level supervision. Our model recovers the higher level structural decomposition of various objects in the form of a binary tree of primitives, where simple parts are represented with fewer primitives and more complex parts are modeled with more components. Our experiments on the ShapeNet and D-FAUST datasets demonstrate that considering the organization of parts indeed facilitates reasoning about 3D geometry.", None),
            Link("Project page", "http:superquadrics.com/hierarchical_primitives", None, None),
            Link("Paper", "https://arxiv.org/pdf/2004.01176.pdf", None, None),
            Link("Poster", "data/Paschalidou2020CVPR_poster.pdf", None, None),
            Link("Code", "https://github.com/paschalidoud/hierarchical_primitives", None, None),
            Link("Blog", "https://autonomousvision.github.io/hierarchical-primitives/", None, None),
            Link("Slides", "http://www.cvlibs.net/publications/Paschalidou2020CVPR_slides.pdf", None, None),
            Link("Video", "https://www.youtube.com/watch?v=QgD0NHbWVlU&vq=hd1080&autoplay=1", None, None),
            Link("Bibtex", None, None, """@InProceedings{Paschalidou2020CVPR,
    title = {Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image},
    author = {Paschalidou, Despoina and Luc van Gool and Geiger, Andreas},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    month = jun,
    year = {2020},
}""")
        ]
    ),
    Paper(
        "Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids",
        # "https://arxiv.org/pdf/1904.09970.pdf",
        "http:superquadrics.com",
        "teasers/superquadrics_revisited.png",
        author_list(authors, "despi", "osman", "andreas"),
        conferences["cvpr"],
        2019,
        None,
        [
            Link("Abstract", None, "Abstracting complex 3D shapes with parsimonious part-based representations has been a long standing goal in computer vision. This paper presents a learning-based solution to this problem which goes beyond the traditional 3D cuboid representation by exploiting superquadrics as atomic elements. We demonstrate that superquadrics lead to more expressive 3D scene parses while being easier to learn than 3D cuboid representations. Moreover, we provide an analytical solution to the Chamfer loss which avoids the need for computational expensive reinforcement learning or iterative prediction. Our model learns to parse 3D objects into consistent superquadric representations without supervision. Results on various ShapeNet categories as well as the SURREAL human body dataset demonstrate the flexibility of our model in capturing fine details and complex poses that could not have been modelled using cuboids.", None),
            Link("Project page", "http:superquadrics.com/learnable-superquadrics.html", None, None),
            Link("Paper", "https://arxiv.org/pdf/1904.09970.pdf", None, None),
            Link("Poster", "data/Paschalidou2019CVPR_poster.pdf", None, None),
            Link("Code", "https://github.com/paschalidoud/superquadric_parsing", None, None),
            Link("Blog", "https://autonomousvision.github.io/superquadrics-revisited/", None, None),
            Link("Video", "https://www.youtube.com/watch?v=eaZHYOsv9Lw", None, None),
            Link("Bibtex", None, None, """@InProceedings{Paschalidou2019CVPR,
    title = {Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids},
    author = {Paschalidou, Despoina and Ulusoy, Ali Osman and Geiger, Andreas},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    month = jun,
    year = {2019},
}""")
        ]
    ),
    Paper(
        "PointFlowNet: Learning Representations for Rigid Motion Estimation from Point Clouds",
        "https://avg.is.tuebingen.mpg.de/publications/behl2019cvpr",
        # "http://www.cvlibs.net/publications/Behl2019CVPR.pdf",
        "teasers/pointflownet.png",
        author_list(authors, "aseem", "despi", "simon", "andreas"),
        conferences["cvpr"],
        2019,
        None,
        [
            Link("Abstract", None, "Despite significant progress in image-based 3D scene flow estimation, the performance of such approaches has not yet reached the fidelity required by many applications. Simultaneously, these applications are often not restricted to image-based estimation: laser scanners provide a popular alternative to traditional cameras, for example in the context of self-driving cars, as they directly yield a 3D point cloud. In this paper, we propose to estimate 3D motion from such unstructured point clouds using a deep neural network. In a single forward pass, our model jointly predicts 3D scene flow as well as the 3D bounding box and rigid body motion of objects in the scene. While the prospect of estimating 3D scene flow from unstructured point clouds is promising, it is also a challenging task. We show that the traditional global representation of rigid body motion prohibits inference by CNNs, and propose a translation equivariant representation to circumvent this problem. For training our deep network, a large dataset is required. Because of this, we augment real scans from KITTI with virtual objects, realistically modeling occlusions and simulating sensor noise. A thorough comparison with classic and learning-based techniques highlights the robustness of the proposed approach.", None),
            # Link("Project page", "https://avg.is.tuebingen.mpg.de/publications/behl2019cvpr", None, None),
            Link("Paper", "http://www.cvlibs.net/publications/Behl2019CVPR.pdf", None, None),
            Link("Code", "https://github.com/aseembehl/pointflownet", None, None),
            Link("Video", "https://youtu.be/cjJhzYCUNTY", None, None),
            Link("Bibtex", None, None, """@InProceedings{Behl2019CVPR,
    title = {PointFlowNet: Learning Representations for Rigid Motion Estimation from Point Clouds },
    author = {Behl, Aseem and Paschalidou, Despoina and Donne, Simon and Geiger, Andreas},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    month = jun,
    year = {2019},
}""")
        ]
    ),
    Paper(
        "RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials",
        # "http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf",
        "http://raynet-mvs.com",
        "teasers/raynet.png",
        author_list(authors, "despi", "osman", "caro", "luc", "andreas"),
        conferences["cvpr"],
        2018,
        "Spotlight Presentation",
        [
            Link("Abstract", None, "In this paper, we consider the problem of reconstructing a dense 3D model using images captured from different views. Recent methods based on convolutional neural networks (CNN) allow learning the entire task from data. However, they do not incorporate the physics of image formation such as perspective geometry and occlusion. Instead, classical approaches based on Markov Random Fields (MRF) with ray-potentials explicitly model these physical processes, but they cannot cope with large surface appearance variations across different viewpoints. In this paper, we propose RayNet, which combines the strengths of both frameworks. RayNet integrates a CNN that learns view-invariant feature representations with an MRF that explicitly encodes the physics of perspective projection and occlusion. We train RayNet end-to-end using empirical risk minimization. We thoroughly evaluate our approach on challenging real-world datasets and demonstrate its benefits over a piece-wise trained baseline, hand-crafted models as well as other learning-based approaches.", None),
            # Link("Project page", "http://raynet-mvs.com", None, None),
            Link("Paper", "http://openaccess.thecvf.com/content_cvpr_2018/papers/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.pdf", None, None),
            Link("Poster", "data/Paschalidou2018CVPR_poster.pdf", None, None),
            Link("Code", "http://github.com/paschalidoud/raynet", None, None),
            Link("Video", "https://www.youtube.com/watch?v=PZ0u1VZLLkU&feature=youtu.be", None, None),
            Link("Bibtex", None, None, """@InProceedings{Paschalidou2018CVPR,
      title = {RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials},
      author = {Paschalidou, Despoina and Ulusoy, Ali Osman and Schmitt, Carolin and Gool, Luc and Geiger, Andreas},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = jun,
      year = {2018}
}""")
        ]
    ),
    Paper(
        "Learning local feature aggregation functions with backpropagation",
        "https://arxiv.org/pdf/1706.08580.pdf",
        "teasers/local_feature_aggregation.png",
        author_list(authors, "despi", "angelos", "diou", "delo"),
        conferences["eusipco"],
        2017,
        None,
        [
            Link("Abstract", None, "This paper introduces a family of local feature aggregation functions and a novel method to estimate their parameters, such that they generate optimal representations for classification (or any task that can be expressed as a cost function minimization problem). To achieve that, we compose the local feature aggregation function with the classifier cost function and we backpropagate the gradient of this cost function in order to update the local feature aggregation function parameters. Experiments on synthetic datasets indicate that our method discovers parameters that model the class-relevant information in addition to the local feature space. Further experiments on a variety of motion and visual descriptors, both on image and video datasets, show that our method outperforms other state-of-the-art local feature aggregation functions, such as Bag of Words, Fisher Vectors and VLAD, by a large margin.", None),
            Link("Paper", "https://arxiv.org/pdf/1706.08580.pdf", None, None),
            Link("Poster", "data/eusipco_poster.pdf", None, None),
            Link("Code", "https://github.com/paschalidoud/feature-aggregation", None, None),
            Link("Bibtex", None, None, """@InProceedings{katharopoulos2017learning
      title = {Learning local feature aggregation functions with backpropagation},
      author = {Paschalidou, Despoina and Katharopoulos, Angelos and Diou, Christos and Delopoulos, Anastasios},
      publisher = {IEEE},
      month = aug,
      year = {2017},
      url = {http://ieeexplore.ieee.org/Abstract/document/8081307/},
}""")
        ]
    ),
    Paper(
        "Fast Supervised LDA for Discovering Micro-Events in Large-Scale Video Datasets",
        "http://ldaplusplus.com", # "https://mug.ee.auth.gr/wp-content/uploads/fsLDA.pdf",
        "teasers/fslda.png",
        author_list(authors, "angelos", "despi", "diou", "delo"),
        conferences["acmmm"],
        2016,
        None,
        [
            Link("Abstract", None, "This paper introduces fsLDA, a fast variational inference method for supervised LDA, which overcomes the computational limitations of the original supervised LDA and enables its application in large-scale video datasets. In addition to its scalability, our method also overcomes the drawbacks of standard, unsupervised LDA for video, including its focus on dominant but often irrelevant video information (e.g. background, camera motion). As a result, experiments in the UCF11 and UCF101 datasets show that our method consistently outperforms unsupervised LDA in every metric. Furthermore, analysis shows that class-relevant topics of fsLDA lead to sparse video representations and encapsulate high-level information corresponding to parts of video events, which we denote 'micro-events'", None),
            # Link("Project page", "http://ldaplusplus.com", None, None),
            Link("Paper", "https://mug.ee.auth.gr/wp-content/uploads/fsLDA.pdf", None, None),
            Link("Poster", "data/fslda_poster.pdf", None, None),
            Link("Code", "https://github.com/angeloskath/supervised-lda", None, None),
            Link("Blog", "https://mug.ee.auth.gr/discovering-micro-events-video-data-using-topic-modeling/", None, None),
            Link("Bibtex", None, None, """@InProceedings{katharopoulos2016fast
        title = {Fast Supervised LDA for Discovering Micro-Events in Large-Scale Video Datasets},
        author = {Katharopoulos, Angelos and Paschalidou, Despoina and Diou, Christos and Delopoulos, Anastasios},
        booktitle = {Proceedings of the 2016 ACM on Multimedia Conference},
        pages = {332,336},
        month = oct,
        year = {2016},
        url = {http://dl.acm.org/citation.cfm?id=2967237},
        month_numeric = {10}

}""")
        ]
    ),
]


def build_publications_list(publications):
    def image(paper):
        if paper.image is not None:
            return '<img src="{}" alt="{}" />'.format(
                paper.image, paper.title
            )
        else:
            return '&nbsp;'

    def title(paper):
        return '<a href="{}">{}</a>'.format(paper.url, paper.title)

    def authors(paper):
        def author(author):
            if author.is_me:
                return '<strong class="author">{}</strong>'.format(author.name)
            else:
                return '<a href="{}" class="author">{}</a>'.format(
                    author.url, author.name
                )
        return ", ".join(author(a) for a in paper.authors)

    def conference(paper):
        cf = '{}, {}'.format(paper.conference.name, paper.year)
        if paper.special is not None:
            cf = cf + '<div class="special">   ({})</div>'.format(paper.special)
        return cf

    def links(paper):
        def links_list(paper):
            def link(i, link):
                if link.url is not None:
                    # return '<a href="{}">{}</a>'.format(link.url, link.name)
                    return '<a href="{}" data-type="{}">{}</a>'.format(link.url, link.name, link.name)
                else:
                    return '<a href="#" data-type="{}" data-index="{}">{}</a>'.format(link.name, i, link.name)
            return " ".join(
                link(i, l) for i, l in enumerate(paper.links)
            )

        def links_content(paper):
            def content(i, link):
                if link.url is not None:
                    return ""
                return '<div class="link-content" data-index="{}">{}</div>'.format(
                    i, link.html if link.html is not None
                       else '<pre>' + link.text + "</pre>"
                )
            return "".join(content(i, link) for i, link in enumerate(paper.links))
        return links_list(paper) + links_content(paper)

    def paper(p):
        return ('<div class="row paper">'
                    '<div class="image">{}</div>'
                    '<div class="content">'
                        '<div class="paper-title">{}</div>'
                        '<div class="conference">{}</div>'
                        '<div class="authors">{}</div>'
                        '<div class="links">{}</div>'
                    '</div>'
                '</div>').format(
                    image(p),
                    title(p),
                    conference(p),
                    authors(p),
                    links(p)
                )

    return "".join(paper(p) for p in publications)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Create a publication list and insert in into an html file"
    )
    parser.add_argument(
        "file",
        help="The html file to insert the publications to"
    )

    parser.add_argument(
        "--safe", "-s",
        action="store_true",
        help="Do not overwrite the file but create one with suffix .new"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Do not output anything to stdout/stderr"
    )

    args = parser.parse_args(argv)

    # Read the file
    with open(args.file) as f:
        html = f.read()

    # Find the fence comments
    start_text = "<!-- start publication list -->"
    end_text = "<!-- end publication list -->"
    start = html.find(start_text)
    end = html.find(end_text, start)
    if end < start or start < 0:
        _print(args, "Could not find the fence comments", file=sys.stderr)
        sys.exit(1)

    # Build the publication list in html
    replacement = build_publications_list(publications)

    # Update the html and save it
    html = html[:start+len(start_text)] + replacement + html[end:]

    # If safe is set do not overwrite the input file
    if args.safe:
        with open(args.file + ".new", "w") as f:
            f.write(html)
    else:
        with open(args.file, "w") as f:
            f.write(html)


if __name__ == "__main__":
    main(None)
