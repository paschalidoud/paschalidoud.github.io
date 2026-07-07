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
    "gordon": Author("Gordon Wetzstein", "https://stanford.edu/~gordonwz/"),
    "konstantinos": Author("Konstantinos Tertikas", "https://ktertikas.github.io/"),
    "boxiao": Author("Boxiao Pan", "https://cs.stanford.edu/~bxpan/"),
    "zhen": Author("Zhen Wang", "https://zhenwangwz.github.io/"),
    "shijie": Author("Shijie Zhou", "https://www.linkedin.com/in/shijie-zhou-ucla"),
    "sherwin": Author("Sherwin Bahmani", "https://sherwinbahmani.github.io/"),
    "xingguang": Author("Xingguang Yan", "http://yanxg.art/"),
    "andrea": Author("Andrea Tagliasacchi", "https://taiya.github.io/"),
    "jj": Author("Jeong Joon Park", ""),
    "achuta": Author("Achuta Kadambi", "https://www.ee.ucla.edu/achuta-kadambi/"),
    "mika": Author("Mikaela Angelina Uy", ""),
    "ziyu": Author("Ziyu Wan", "http://raywzy.com/"),
    "ian": Author("Ian Huang", "https://ianhuang0630.github.io/me/"),
    "hongyu": Author("Hongyu Liu", "https://kumapowerliu.github.io/"),
    "xiaoyu": Author("Xiaoyu Xiang", "https://engineering.purdue.edu/people/xiaoyu.xiang.1"),
    "jing": Author("Jing Liao", ""),
    "davis": Author("Davis Rempe", "https://davrempe.github.io/"),
    "colton": Author("Colton Stearns", "https://coltonstearns.github.io/"),
    "jiateng": Author("Jiateng Liu", "https://lumos-jiateng.github.io/"),
    "alex": Author("Alex Fu", ""),
    "sebastien": Author("Sébastien Mascha", "https://github.com/sebastienmascha"),
    "will": Author("Bokui Shen", ""),
    "suya": Author("Suya You", ""),
    "yannis": Author("Yannis Avrithis", "https://avrithis.net/"),
    "emiris": Author("Ioannis Emiris", "https://cgi.di.uoa.gr/~emiris/index-eng.html"),
    "amlan": Author("Amlan Kar", "https://amlankar.github.io/"),
    "masha": Author("Maria Shugrina", "http://shumash.com/"),
    "karsten": Author("Karsten Kreis", "https://scholar.google.de/citations?user=rFd-DiAAAAAJ&hl=de"),
    "aseem": Author("Aseem Behl", "http://aseembehl.github.io/"),
    "simon": Author("Simon Donné", "https://donnessime.github.io/"),
    "caro": Author("Carolin Schmitt", "https://avg.is.tuebingen.mpg.de/person/cschmitt"),
    "luc": Author("Luc van Gool", "https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html"),
    "angelos": Author("Angelos Katharopoulos", "https://angeloskath.github.io/"),
    "diou": Author("Christos Diou", "https://mug.ee.auth.gr/people/christos-diou/"),
    "delo": Author("Anastasios Delopoulos", "https://mug.ee.auth.gr/people/anastasios-delopoulos/"),
    "xindi": Author("Xindi Wu", "https://xindiwu.github.io/"),
    "jungao": Author("Jun Gao", "https://www.cs.toronto.edu/~jungao/"),
    "torralba": Author("Antonio Torralba", "https://groups.csail.mit.edu/vision/torralbalab/"),
    "laura": Author("Laura Leal-Taixé", "https://dvl.in.tum.de/team/lealtaixe/"),
    "olga": Author("Olga Russakovsky", "https://www.cs.princeton.edu/~olgarus/"),
    "lorraine": Author("Jonathan Lorraine", "https://www.jonlorraine.com/"),
    "aarti": Author("Aarti Basant", "https://www.linkedin.com/in/aartibasant/"),
    "fangyin": Author("Fangyin Wei", "https://weify627.github.io/"),
    "ferroni": Author("Francesco Ferroni", "https://www.francescoferroni.com/"),
    "guillermo": Author("Guillermo Garcia Cobo", "https://scholar.google.com/citations?user=zdWIO6cAAAAJ&hl=en"),
    "haithem": Author("Haithem Turki", "https://haithemturki.com/"),
    "huanling": Author("Huan Ling", "https://www.cs.toronto.edu/~linghuan/"),
    "jaewoo": Author("Jaewoo Seo", "https://scholar.google.com/citations?user=3IOC9IsAAAAJ&hl=en"),
    "james": Author("James Lucas", "https://www.cs.toronto.edu/~jlucas/"),
    "jay": Author("Jay Zhangjie Wu", "https://zhangjiewu.github.io/"),
    "jialiang": Author("Jialiang Wang", "https://sites.google.com/view/jialiangwang/home"),
    "kaihe": Author("Kai He", "https://www.cs.toronto.edu/~hekai/"),
    "katarina": Author("Katarina Tothova", "https://scholar.google.com/citations?user=tua-w_UAAAAJ&hl=en"),
    "kevinxie": Author("Kevin Xie", "https://kevincxie.github.io/"),
    "michal": Author("Michał Tyszkiewicz", "https://scholar.google.com/citations?user=CZ40rFYAAAAJ&hl=en"),
    "qiwu": Author("Qi Wu", "https://wilsoncernwq.github.io/"),
    "riccardo": Author("Riccardo de Lutio", "https://riccardodelutio.github.io/"),
    "ruilong": Author("Ruilong Li", "https://www.liruilong.cn/"),
    "seung": Author("Seung Wook Kim", "https://seung-kim.github.io/seungkim/"),
    "tianchang": Author("Tianchang Shen", "https://www.cs.toronto.edu/~shenti11/"),
    "tianshi": Author("Tianshi Cao", "https://scholar.google.com/citations?user=CZ9wBBoAAAAJ&hl=en"),
    "tobias": Author("Tobias Pfaff", "https://tobiaspfaff.com/"),
    "williamlew": Author("William Lew", "https://www.linkedin.com/in/williamlewww/"),
    "xuanchi": Author("Xuanchi Ren", "https://xuanchiren.com/"),
    "yifan": Author("Yifan Lu", "https://yifanlu0227.github.io/"),
    "yuxuan": Author("Yuxuan Zhang", "https://scholar.google.com/citations?user=Jt5VvNgAAAAJ&hl=en"),
    "zan": Author("Zan Gojcic", "https://zgojcic.github.io/"),
    "zian": Author("Zian Wang", "https://www.cs.toronto.edu/~zianwang/"),
    "songlin": Author("Songlin Li", "")
}
conferences = {
    "neurips": Conference("Advances in Neural Information Processing Systems (NeurIPS)"),
    "cvpr": Conference("Computer Vision and Pattern Recognition (CVPR)"),
    "iccv": Conference("International Conference on Computer Vision (ICCV)"),
    "eusipco": Conference("European Signal Processing Conference (EUSIPCO)"),
    "acmmm": Conference("ACM Multimedia Conference (ACMM)"),
    "icml": Conference("International Conference on Machine Learning (ICML)"),
    "threedv": Conference("International Conference on 3D Vision (3DV)"),
    "arxiv": Conference("arXiv")
}
publications = [
    Paper(
        "NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation",
        "https://research.nvidia.com/labs/sil/projects/omnidreams-blog/",
        "teasers/omnidreams_teaser.jpg",
        author_list(authors, "aarti", "amlan", "despi", "fangyin", "ferroni", "guillermo", "haithem", "huanling", "jaewoo", "james", "jay", "jialiang", "lorraine", "jungao", "kaihe", "katarina", "kevinxie", "michal", "qiwu", "riccardo", "ruilong", "sanja", "seung", "tianchang", "tianshi", "tobias", "williamlew", "xindi", "xuanchi", "yifan", "yuxuan", "zan", "zian"),
        conferences["arxiv"],
        2026,
        None,
        [   Link("Abstract", None, "As autonomous vehicle capabilities advance, the safe evaluation of driving policies in long-tail scenarios remains a critical bottleneck. In closed-loop simulation, the driving policy model actively interacts with the environment, where its actions dynamically update the simulator state and directly influence the next set of generated sensor observations. While recent reconstruction-based neural simulators offer photorealism, they are fundamentally constrained by their initial captured data and struggle to generalize to highly dynamic or novel scenes. To overcome these limitations, we introduce OmniDreams, a foundation generative world model mid- and post-trained from the Cosmos diffusion model to autoregressively generate action-conditioned videos in real time. By leveraging the rich visual priors of Cosmos and mid- and post-training on 21k hours of driving scenarios, OmniDreams synthesizes complex, unobserved phenomena that are hard for traditional simulators to capture, such as extreme weather and unpredictable dynamic agent behaviors. Crucially, it autoregressively conditions its photorealistic sensor generation on past frames, the current simulator state, and immediate driving actions. Deployed in a closed-loop system with the Alpamayo 1 policy model and AlpaSim orchestrator, OmniDreams acts as a highly responsive, reactive environment, providing a scalable and comprehensive solution for training and evaluating next-generation autonomous driving policies. We additionally show preliminary results indicating that a world-action model (WAM) post-trained from OmniDreams achieves strong performance on the Physical AI Autonomous Vehicles NuRec dataset, surpassing the VLA-based Alpamayo 1.5 research policy model while using only 1/5 the total parameters. These results highlight the potential for a real-time world model like OmniDreams to also serve as a backbone for policy architectures.", None),
            Link("Project page", "https://research.nvidia.com/labs/sil/projects/omnidreams-blog/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2606.03159", None, None),
            Link("Code", "https://github.com/nv-tlabs/omni-dreams", None, None),
            Link("Bibtex", None, None, """@article{Basant2026arXiv,
      title={NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation},
      author={Basant, Aarti and Kar, Amlan and Paschalidou, Despoina and Wei, Fangyin and Ferroni, Francesco and Garcia Cobo, Guillermo and Turki, Haithem and Ling, Huan and Seo, Jaewoo and Lucas, James and Wu, Jay Zhangjie and Wang, Jialiang and Lorraine, Jonathan and Gao, Jun and He, Kai and Tothova, Katarina and Xie, Kevin and Tyszkiewicz, Micha\\l{} and Wu, Qi and de Lutio, Riccardo and Li, Ruilong and Fidler, Sanja and Kim, Seung Wook and Shen, Tianchang and Cao, Tianshi and Pfaff, Tobias and Lew, William and Wu, Xindi and Ren, Xuanchi and Lu, Yifan and Zhang, Yuxuan and Gojcic, Zan and Wang, Zian},
      journal={arXiv preprint arXiv:2606.03159},
      year={2026}
    }
""")
        ]
    ),

    Paper(
        "Motion Attribution for Video Generation",
        "https://research.nvidia.com/labs/sil/projects/MOTIVE/",
        "teasers/motive_teaser.png",
        author_list(authors, "xindi", "despi", "jungao", "torralba", "laura", "olga", "sanja", "lorraine"),
        conferences["icml"],
        2026,
        "Oral, Honorable Mention for Outstanding Paper",
        [   Link("Abstract", None, "Despite the rapid progress of video generation models, the role of data in influencing motion is poorly understood. We present Motive, a motion-centric, gradient-based data attribution framework that scales to modern, large, high-quality video datasets and models, and use it to study which fine-tuning clips improve or degrade temporal dynamics. Our approach isolates temporal dynamics from static appearance via motion-weighted loss masks, yielding efficient and scalable motion-specific influence computation. On text-to-video models, Motive identifies clips that strongly affect motion and guides data curation that improves temporal consistency and physical plausibility. With Motive-selected high-influence data, our method improves both motion smoothness and dynamic degree on VBench, achieving a 74.1% human preference win rate compared with the pretrained base model. This represents the first framework to attribute motion rather than visual appearance in video generative models and to use it to curate fine-tuning data.", None),
            Link("Project page", "https://research.nvidia.com/labs/sil/projects/MOTIVE/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2601.08828", None, None),
            Link("Slides", "https://www.canva.com/design/DAG1yWt_m7M/_exm2P38U7t_TGoN8N2YdQ/view?utm_content=DAG1yWt_m7M&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h0a72b7c054", None, None),
            Link("Poster", "https://research.nvidia.com/labs/sil/projects/MOTIVE/poster/poster-horizontal.pdf", None, None),
            Link("Video", "https://www.youtube.com/watch?v=oMimFdRu39U", None, None),
            Link("Bibtex", None, None, """@inproceedings{Wu2026ICML,
      title={Motion Attribution for Video Generation},
      author={Wu, Xindi and Paschalidou, Despoina and Gao, Jun and Torralba, Antonio and Leal-Taix\\'e, Laura and Russakovsky, Olga and Fidler, Sanja and Lorraine, Jonathan},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2026}
    }
""")
        ]
    ),

    Paper(
        "PASTA: Controllable Part-Aware Shape Generation with Autoregressive Transformers",
        "https://arxiv.org/abs/2407.13677",
        "teasers/pasta_teaser.jpeg",
        author_list(authors, "songlin", "despi", "leo"),
        conferences["threedv"],
        2026,
        None,
        [   Link("Abstract", None, "The increased demand for tools that automate the 3D content creation process led to tremendous progress in deep generative models that can generate diverse 3D objects of high fidelity. In this paper, we present PASTA, an autoregressive transformer architecture for generating high quality 3D shapes. PASTA comprises two main components: An autoregressive transformer that generates objects as a sequence of cuboidal primitives and a blending network, implemented with a transformer decoder that composes the sequences of cuboids and synthesizes high quality meshes for each object. Our model is trained in two stages: First we train our autoregressive generative model using only annotated cuboidal parts as supervision and next, we train our blending network using explicit 3D supervision, in the form of watertight meshes. Evaluations on various ShapeNet objects showcase the ability of our model to perform shape generation from diverse inputs e.g. from scratch, from a partial object, from text and images, as well size-guided generation, by explicitly conditioning on a bounding box that defines the object's boundaries. Moreover, as our model considers the underlying part-based structure of a 3D object, we are able to select a specific part and produce shapes with meaningful variations of this part. As evidenced by our experiments, our model generates 3D shapes that are both more realistic and diverse than existing part-based and non part-based methods, while at the same time is simpler to implement and train.", None),
            Link("Paper", "https://arxiv.org/pdf/2407.13677", None, None),
            Link("Bibtex", None, None, """@inproceedings{Li20263DV,
      title={PASTA: Controllable Part-Aware Shape Generation with Autoregressive Transformers},
      author={Li, Songlin and Paschalidou, Despoina and Guibas, Leonidas},
      booktitle={International Conference on 3D Vision (3DV)},
      year={2026}
    }
""")
        ]
    ),

    Paper(
        "Cosmos World Foundation Model Platform for Physical AI",
        "https://www.nvidia.com/en-us/ai/cosmos/",
        "teasers/cosmos_teaser.png",
        [
            Author('Niket Agarwal', 'https://www.linkedin.com/in/niket-agarwal-9522b27/'),
            Author('Arslan Ali', 'https://scholar.google.com/citations?user=P4QdAtoAAAAJ&hl=en'),
            Author('Maciej Bala', 'https://www.linkedin.com/in/maciej-bala-0a7b3a12b/'),
            Author('Yogesh Balaji', 'https://yogeshbalaji.github.io/'),
            Author('Erik Barker', ''),
            Author('Tiffany Cai', ''),
            Author('Prithvijit Chattopadhyay', 'https://prithv1.github.io/'),
            Author('Yongxin Chen', 'https://yongxin.ae.gatech.edu/'),
            Author('Yin Cui', 'https://ycui.me/'),
            Author('Yifan Ding', ''),
            Author('Daniel Dworakowski', 'https://danieldworakowski.github.io/'),
            Author('Jiaojiao Fan', 'https://sbyebss.github.io/'),
            Author('Michele Fenzi', 'https://scholar.google.com/citations?user=x3xLe8wAAAAJ&hl=en'),
            Author('Francesco Ferroni', 'https://www.francescoferroni.com/'),
            Author('Sanja Fidler', 'https://www.cs.utoronto.ca/~fidler/'),
            Author('Dieter Fox', 'https://homes.cs.washington.edu/~fox/'),
            Author('Songwei Ge', 'https://songweige.github.io/'),
            Author('Yunhao Ge', 'https://gyhandy.github.io/'),
            Author('Jinwei Gu', 'https://www.gujinwei.org/'),
            Author('Siddharth Gururani', 'https://scholar.google.com/citations?user=_C-H8_MAAAAJ&hl=en'),
            Author('Ethan He', 'https://ethanhe.ai'),
            Author('Jiahui Huang', 'https://huangjh-pub.github.io/'),
            Author('Jacob Huffman', ''),
            Author('Pooya Jannaty', 'https://www.linkedin.com/in/pooyaj/'),
            Author('Jingyi Jin', 'https://www.linkedin.com/in/jingyi-jin/'),
            Author('Seung Wook Kim', 'https://seung-kim.github.io/seungkim/'),
            Author('Gergely Klár', 'https://scholar.google.com/citations?hl=en&user=WuKqNrgAAAAJ'),
            Author('Grace Lam', 'https://github.com/grace-lam'),
            Author('Shiyi Lan', 'https://scholar.google.com/citations?user=jIUI6F4AAAAJ&hl=en'),
            Author('Laura Leal-Taixé', 'https://dvl.in.tum.de/team/lealtaixe/'),
            Author('Anqi Li', 'https://anqili.github.io/'),
            Author('Zhaoshuo Li', 'https://mli0603.github.io/'),
            Author('Chen-Hsuan Lin', 'https://chenhsuanlin.bitbucket.io/'),
            Author('Tsung-Yi Lin', 'https://tsungyilin.info/'),
            Author('Huan Ling', 'https://www.cs.toronto.edu/~linghuan/'),
            Author('Ming-Yu Liu', 'https://mingyuliu.net/'),
            Author('Xian Liu', 'https://alvinliu0.github.io/'),
            Author('Alice Luo', 'https://www.linkedin.com/in/aliceluoqian/'),
            Author('Qianli Ma', 'https://qianlim.github.io/'),
            Author('Hanzi Mao', 'https://hanzimao.me/'),
            Author('Kaichun Mo', 'https://cs.stanford.edu/~kaichun/'),
            Author('Arsalan Mousavian', 'https://scholar.google.com/citations?user=fcA9m88AAAAJ&hl=en'),
            Author('Seungjun Nah', 'https://seungjunnah.github.io/'),
            Author('Sriharsha Niverty', 'https://www.linkedin.com/in/sriharsha-niverty-a2412818'),
            Author('David Page', ''),
            Author('Despoina Paschalidou', ''),
            Author('Zeeshan Patel', 'https://www.zeeshanp.me/'),
            Author('Lindsey Pavao', 'https://www.linkedin.com/in/lindseypavao/'),
            Author('Morteza Ramezanali', 'https://www.linkedin.com/in/xdreamer/'),
            Author('Fitsum Reda', 'https://fitsumreda.github.io/'),
            Author('Xiaowei Ren', 'https://ericrxw.github.io/xiaoweiren/'),
            Author('Vasanth Rao Naik Sabavat', ''),
            Author('Ed Schmerling', 'https://research.nvidia.com/person/ed-schmerling'),
            Author('Stella Shi', 'https://www.linkedin.com/in/yaoshi-ys/'),
            Author('Bartosz Stefaniak', ''),
            Author('Shitao Tang', 'https://tangshitao.github.io/'),
            Author('Lyne Tchapmi', 'https://scholar.google.com/citations?user=1nLxyYcAAAAJ&hl=en'),
            Author('Przemek Tredak', 'https://github.com/ptrendx'),
            Author('Wei-Cheng Tseng', 'https://weichengtseng.github.io/'),
            Author('Jibin Varghese', 'https://codejrv.github.io/'),
            Author('Hao Wang', ''),
            Author('Haoxiang Wang', 'https://haoxiang-wang.github.io/'),
            Author('Heng Wang', 'https://hengcv.github.io/'),
            Author('Ting-Chun Wang', 'https://tcwang0509.github.io/'),
            Author('Fangyin Wei', 'https://weify627.github.io/'),
            Author('Xinyue Wei', 'https://sarahweiii.github.io/'),
            Author('Jay Zhangjie Wu', 'https://zhangjiewu.github.io/'),
            Author('Jiashu Xu', 'https://cnut1648.github.io/'),
            Author('Wei Yang', 'https://wyang.me/'),
            Author('Lin Yen-Chen', 'https://yenchenlin.me/'),
            Author('Xiaohui Zeng', 'https://www.cs.utoronto.ca/~xiaohui/'),
            Author('Yu Zeng', 'https://zengxianyu.github.io/'),
            Author('Jing Zhang', 'https://research.nvidia.com/person/jing-zhang'),
            Author('Qinsheng Zhang', 'https://qsh-zh.github.io/'),
            Author('Yuxuan Zhang', 'https://scholar.google.com/citations?user=Jt5VvNgAAAAJ&hl=en'),
            Author('Qingqing Zhao', 'https://scholar.google.com/citations?user=UVMgJvYAAAAJ&hl=en'),
            Author('Artur Zolkowski', 'https://ch.linkedin.com/in/azolkowski'),
        ],
        conferences["arxiv"],
        2025,
        None,
        [   Link("Abstract", None, 'Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make Cosmos open-source and our models open-weight with permissive licenses.', None),
            Link("Project page", "https://www.nvidia.com/en-us/ai/cosmos/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2501.03575", None, None),
            Link("Code", "https://github.com/NVIDIA/Cosmos", None, None),
            Link("Video", "https://www.youtube.com/watch?v=9Uch931cDx8", None, None),
            Link("Bibtex", None, None, """@article{Agarwal2025arXiv,
      title={Cosmos World Foundation Model Platform for Physical AI},
      author={NVIDIA and Agarwal, Niket and Ali, Arslan and Bala, Maciej and Balaji, Yogesh and others},
      journal={arXiv preprint arXiv:2501.03575},
      year={2025}
    }
""")
        ]
    ),

    Paper(
        "CAD: Photorealistic 3D Generation via Adversarial Distillation",
        "http://raywzy.com/CAD/",
        "teasers/cad_teaser_2.png",
        author_list(authors, "ziyu", "despi", "ian", "hongyu", "will", "xiaoyu", "jing", "leo"),
        conferences["cvpr"],
        2024,
        None,
        [   Link("Abstract", None, "The increased demand for 3D data in AR/VR, robotics and gaming applications, gave rise to powerful generative pipelines capable of synthesizing high-quality 3D objects. Most of these models rely on the Score Distillation Sampling (SDS) algorithm to optimize a 3D representation such that the rendered image maintains a high likelihood as evaluated by a pre-trained diffusion model. However, finding a correct mode in the high-dimensional distribution produced by the diffusion model is challenging and often leads to issues such as over-saturation, over-smoothing, and Janus-like artifacts. In this paper, we propose a novel learning paradigm for 3D synthesis that utilizes pre-trained diffusion models. Instead of focusing on mode-seeking, our method directly models the distribution discrepancy between multi-view renderings and diffusion priors in an adversarial manner, which unlocks the generation of high-fidelity and photorealistic 3D content, conditioned on a single image and prompt. Moreover, by harnessing the latent space of GANs and expressive diffusion model priors, our method facilitates a wide variety of 3D applications including single-view reconstruction, high diversity generation and continuous 3D interpolation in the open domain. The experiments demonstrate the superiority of our pipeline compared to previous works in terms of generation quality and diversity.", None),
            Link("Project page", "http://raywzy.com/CAD/", None, None),
            Link("Paper", "https://arxiv.org/pdf/2312.06663.pdf", None, None),
            Link("Code", "https://github.com/raywzy/CAD", None, None),
            Link("Video", "https://www.youtube.com/watch?v=6slL9YqW9JM", None, None),
            Link("Bibtex", None, None, """@InProceedings{Wan2024CVPR,
      title={CAD: Photorealistic 3d generation via adversarial distillation}, 
      author={Wan, Ziyu and Paschalidou, Despoina and Huang, Ian and Liu, Hongyu and Shen, Bokui and Xiang, Xiaoyu and Liao, Jing and Guibas, Leonidas},
      booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      year = {2024}
    }
""")
        ]
    ),

    Paper(
        "CurveCloudNet: Processing Point Clouds with 1D Structure",
        "https://arxiv.org/abs/2303.12050",
        "teasers/curvecloudnet_teaser.png",
        author_list(authors, "colton", "davis", "alex", "jiateng", "sebastien", "jj", "despi", "leo"),
        conferences["cvpr"],
        2024,
        None,
        [   Link("Abstract", None, "Modern depth sensors such as LiDAR operate by sweeping laser-beams across the scene, resulting in a point cloud with notable 1D curve-like structures. In this work, we introduce a new point cloud processing scheme and backbone, called CurveCloudNet, which takes advantage of the curve-like structure inherent to these sensors. While existing backbones discard the rich 1D traversal patterns and rely on generic 3D operations, CurveCloudNet parameterizes the point cloud as a collection of polylines (dubbed a curve cloud), establishing a local surface-aware ordering on the points. By reasoning along curves, CurveCloudNet captures lightweight curve-aware priors to efficiently and accurately reason in several diverse 3D environments. We evaluate CurveCloudNet on multiple synthetic and real datasets that exhibit distinct 3D size and structure. We demonstrate that CurveCloudNet outperforms both point-based and sparse-voxel backbones in various segmentation settings, notably scaling to large scenes better than point-based alternatives while exhibiting improved single-object performance over sparse-voxel alternatives. In all, CurveCloudNet is an efficient and accurate backbone that can handle a larger variety of 3D environments than past works.", None),
            Link("Paper", "https://arxiv.org/pdf/2303.12050.pdf", None, None),
            Link("Bibtex", None, None, """@InProceedings{Stearns2024CVPR,
      title={CurveCloudNet: Processing Point Clouds with 1D Structure},
      author={Stearns, Colton and Rempe, Davis and Fu, Alex and Liu, Jiateng and Mascha, Sébastien and Park, Jeong Joon and Paschalidou, Despoina and Guibas, Leonidas J},
      booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      year = {2024}
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
            Link("Bibtex", None, None, """@InProceedings{Bahmani2023ICCV,
  author = {Bahmani, Sherwin and Park, Jeong Joon and Paschalidou, Despoina and Yan, Xingguang and Wetzstein, Gordon and Guibas, Leonidas and Tagliasacchi, Andrea},
  title = {CC3D: Layout-Conditioned Generation of Compositional 3D Scenes},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2023}
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
            Link("Video", "https://www.youtube.com/watch?v=H5jJryZzRs8", None, None),
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
        "https://nv-tlabs.github.io/ATISS/",
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
