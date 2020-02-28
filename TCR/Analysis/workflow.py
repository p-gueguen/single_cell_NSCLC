#! /usr/bin/python3

import numpy as np
import pandas as pd
import os
from subprocess import call
import scanpy as sc
from io import StringIO
import subprocess
import sys
import shlex
import editdistance as levenshtein
import networkx as nx
import scipy.cluster.hierarchy
import scipy.spatial
import seaborn as sns


def create_clonotypes(ann, method="greedy_cut"):
    """
    Recreate clonotypes with different methods

    ann: pandas dataframe. Contain at least the columns barcode,
    raw_clonotype_id, nSeqCDR3 and chain)

    method:
    method="greedy":
        Start with the raw_clonotype_id of read10X
        If two cells contain the same nSeqCDR3 they
        share the same clonotype
    method="greedy_dl":
        Only keep links which are found in at least two different cells
    method="greedy_cut":
        Start with the result of "greedy", if a clonotype contains more
        than 2a - 1b, remove any link that's not in at least 2 cells
        and redo it. 
    method="beta":
        Merge together clonotype with the same beta, if they are not too big
        (b1, a1) include (b1), (b1, a2) (but not (a2)) 
    method="pure":
        Only keep exactly identical clonotypes with at least one TRA & one TRB
    """
    if(method == "pure" or method == "beta"):
        seqs_bcode = ann.groupby("barcode").nSeqCDR3.unique().apply(tuple)
        set_to_nb = {v: i for i, v in enumerate(list(set(seqs_bcode.values)))}
        seqs_bcode = seqs_bcode.to_frame().reset_index()
        seqs_bcode["num"] = seqs_bcode.nSeqCDR3.apply(lambda x: set_to_nb[x])
        seqs_bcode["name"] = "clonotype_pure_" + seqs_bcode["num"].apply(str)
        ann["clonotype_pure"] = ann.barcode.map(
            seqs_bcode.set_index("barcode").name)
        ann.loc[ann.clonotype_pure.map(
            (ann.groupby("clonotype_pure").chain.nunique() < 2).to_dict()),
               "clonotype_pure"] = np.NaN
    if(method == "beta"):
        seq_to_cl = ann[ann.chain == "TRB"].groupby(
            "nSeqCDR3").clonotype_pure.first().to_dict()
        ann["clonotype_beta"] = ann.nSeqCDR3.map(seq_to_cl)

    if(method == "greedy"):
        seq_to_cl = ann.groupby("nSeqCDR3").raw_clonotype_id.first().to_dict()
        ann["clonotype_greedy"] = ann.nSeqCDR3.map(seq_to_cl)

    if(method == "greedy_cut" or method == "greedy_dl"):
        # find for a given sequence in how much cells it's in
        seq_to_nbcells = ann.groupby("raw_clonotype_id").barcode.nunique().to_dict()
        ann["nb_cells"] = ann.raw_clonotype_id.map(seq_to_nbcells)
        # find greedy clonotypes
        seq_to_cl = ann.groupby("nSeqCDR3").raw_clonotype_id.first().to_dict()
        ann["clonotype_greedy"] = ann.nSeqCDR3.map(seq_to_cl)
        # find greedy clonotypes by only considering double links
        barcode_to_seq = ann.dropna(subset=["nSeqCDR3"]).groupby("barcode").nSeqCDR3.apply(list).to_dict()
        links = {}
        for b in barcode_to_seq:
            for x in barcode_to_seq[b]:
                for y in barcode_to_seq[b]:
                    if x < y:
                        if (x, y) in links:
                            links[(x, y)] += 1
                        else:
                            links[(x, y)] = 1
        kept_links = {k: v for k, v in links.items() if v > 1}
        G = nx.Graph()
        G.add_nodes_from(set(ann.nSeqCDR3.values))
        G.add_edges_from(kept_links.keys())
        dl_clonotypes = {}
        for i, cc in enumerate(nx.connected_components(G)):
            for seq in cc:
                dl_clonotypes[seq] = "clonotype_dl_" + str(i)
        ann["clonotype_greedy_dl"] = ann.nSeqCDR3.map(dl_clonotypes)

        # mark too big clonotypes
        nb_TRA = (ann[ann.chain == "TRA"].
                  groupby("clonotype_greedy").barcode.nunique().to_dict())
        nb_TRB = (ann[ann.chain == "TRB"].
                  groupby("clonotype_greedy").barcode.nunique().to_dict())
        ann["too_big"] = True
        ann.loc[ann.chain == "TRA",
                "too_big"] = (ann[ann.chain == "TRA"].
                              clonotype_greedy.map(nb_TRA) > 2)
        ann.loc[ann.chain == "TRB",
                "too_big"] = (ann[ann.chain == "TRB"].
                              clonotype_greedy.map(nb_TRB) > 1)
        
        ann.loc[ann.too_big, "clonotype_greedy_cut"] = ann[ann.too_big].clonotype_greedy_dl
        ann.loc[~ann.too_big, "clonotype_greedy_cut"] = ann[~ann.too_big].clonotype_greedy
        ann.drop(columns=["too_big"], inplace=True)

    return ann


def cluster_info(rdata):
    """ From the csv files extracted from the rdata,
    create a  dataframe with all patients, and all barcodes """
    df = pd.read_csv(rdata + "_clusters.csv", index_col="barcode")
    df = df.join(pd.read_csv(rdata + "_metadata.csv", index_col="barcode"))
    df = df.join(pd.read_csv(rdata + "_umap.csv", index_col="barcode"))
    df = df.join(pd.read_csv(rdata + "_pca.csv", index_col="barcode"))
    df["umap"] = df.apply(lambda r: (r["UMAP1"], r["UMAP2"]), axis=1)
    lst_cc = [u for u in df.keys() if "ACC" in u[:4]]
    df["pos_cca"] = df.apply(lambda r: np.array([r[c] for c in lst_cc]),
                                 axis=1)
    df.index.name = "barcode"
    df.reset_index(inplace=True)
    return df


def add_cluster_info(df, cluster_fn, patient, name_column="cluster",
                     metadata=None, umap=None, all_genes=None, cca=None):
    df_clusters = pd.read_csv(cluster_fn)
    df_clusters = df_clusters[df_clusters.barcode.apply(
        lambda x: x.split("_")[1] == patient)]
    df_clusters["cut_barcode"] = df_clusters.barcode.apply(
        lambda x: x.split("_")[0])
    df_clusters = df_clusters.set_index("cut_barcode")
    dct_clusters = df_clusters.cluster.to_dict()
    df["cut_barcode"] = df.barcode.apply(lambda x: x.split("_")[0][:-2])
    df[name_column] = df.cut_barcode.map(dct_clusters)

    if(umap is not None):
        df_pos = pd.read_csv(umap)
        df_pos = df_pos[df_pos.barcode.apply(lambda x:
                                             x.split("_")[1] == patient)]
        df_pos["cut_barcode"] = df_pos.barcode.apply(lambda x: x.split("_")[0])
        df_pos["x"] = df_pos.apply(lambda r: [r["UMAP1"], r["UMAP2"]], axis=1)
        dct_pos = df_pos.set_index("cut_barcode").x.to_dict()
        df["umap"] = df.cut_barcode.map(dct_pos)

    if cca is not None:
        df_pos = pd.read_csv(cca)
        df_pos = df_pos[df_pos.barcode.apply(lambda x:
                                             x.split("_")[1] == patient)]
        df_pos["cut_barcode"] = df_pos.barcode.apply(lambda x: x.split("_")[0])
        df = df.merge(df_pos, on="cut_barcode", how="left",
                      suffixes=("", "_cca"))
        lst_cc = [u for u in df.keys() if "PC" in u[:4]]
        df["pos_cca"] = df.apply(lambda r: np.array([r[c] for c in lst_cc]),
                                 axis=1)

    if(metadata is not None):
        df_data = pd.read_csv(metadata)
        df_data = df_data[df_data.barcode.apply(
            lambda x: x.split("_")[1] == patient)]
        df_data["cut_barcode"] = df_data.barcode.apply(
            lambda x: x.split("_")[0])
        df = df.merge(df_data, on="cut_barcode", how="left",
                      suffixes=("", "_mdata"))
    if(all_genes is not None):
        pass
    return df


def mixcr(sequence_file, output=None, intermediate_repertory="/tmp/"):
    """
    Run mixcr on fastq file (or fasta ?). Don't assemble the contigs.
    """
    if(output is None):
        output = "/tmp/analysis.tsv"
    
    cmd_align = ("mixcr align --force-overwrite -OsaveOriginalReads=true "
                 "-s human {} {}analysis.vdjca").format(
                     sequence_file, intermediate_repertory)
    cmd_export = ("mixcr exportAlignments  --force-overwrite --preset full "
                  "-descrsR1 -readIds -chains -vGene -dGene -jGene -cGene -vFamily -jFamily "
                  "{}analysis.vdjca {}").format(
                      intermediate_repertory, output)
    run(cmd_align)
    run(cmd_export)
    df = pd.read_csv(output, sep="\t")
    return df



def run_mixcr_tcr(directory, filtered=False):
    """
    Run mixcr and add information
    @ Arguments:
    * directory: the path where the outs folder is located
    * filtered: either consider the "filtered_contig" file or "all_contig"
    """
    if not os.path.exists(directory + "mixcr"):
        os.makedirs(directory + "mixcr")

    file_type = "all"
    if filtered:
        file_type = "filtered"

    # first run mixcr on all_contig
    df = mixcr(directory + "outs/"+file_type+"_contig.fastq",
               output=directory+"mixcr/"+file_type+"_contig_mixcr.tsv",
               intermediate_repertory=directory+"mixcr/")
    df.set_index("descrsR1", drop=False, inplace=True)  # index by fasta name

    # open the file all_contig_annotations.csv
    df_ann = pd.read_csv(directory+"outs/"+file_type+"_contig_annotations.csv",
                         sep=",")
    df_ann = df_ann[["contig_id", "reads", "umis",
                     "raw_clonotype_id", "barcode"]]
    df_ann.set_index("contig_id", drop=False, inplace=True)
    
    result = pd.concat([df, df_ann], axis=1, join='inner')
    result.rename(columns={'Chains': 'chain'}, inplace=True)
    result.to_csv(directory+"outs/"+file_type+"_contig_annotations_mixcr.csv")
    return result


def load_pgen(df, repertory):
    """
    get Pgen from the IGoR files
    """
    df_ig = pd.read_csv(repertory + "outs/all_contig_igor.tsv", sep="\t")
    df_ig.set_index("contig_id", drop=False, inplace=True)
    dct_pgen = df_ig.Pgen.to_dict()
    df["pgen"] = df.contig_id.map(dct_pgen)


def runProcess(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            break
        yield line


def run(cmd):
    for l in runProcess(shlex.split(cmd)):
        print(l)



def load_signatures(sign_dir):
    signs = {}
    with open(sign_dir + "list_signatures_files.txt") as fsigns:
        for l in fsigns:
            k, v = l.strip().split(";")
            with open(sign_dir + v) as f:
                signs[k] = [a.strip().upper() for a in f.readlines()]
            
    return signs
    

def add_transcriptome_information(df, adata, cols):
    """ add transcription's information to the TCR dataframe
    Args:
    - df is the TCR dataframe (panda dataframe), with a "barcode" column
    - adata is a AnnData object 
    - cols is a list containing the columns that will be added
    """
    for a in cols:
        df[a] = df.barcode.map(adata.obs[a])
    

def load_transcriptome(transcriptome_matrix):
    adata = sc.read_10x_mtx(
        transcriptome_matrix, var_names='gene_symbols', cache=True)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, min_cells=3)
    adata.var["mito"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, feature_controls=["mito"], inplace=True)
    adata.var["ribo"] = adata.var_names.str.contains("^RPL|^RPS")
    sc.pp.calculate_qc_metrics(adata, feature_controls=["ribo"], inplace=True)
    adata = adata[adata.obs['total_features_by_counts'] < 5000, :]
    adata = adata[adata.obs['pct_counts_mito'] < 10, :]
    adata = adata[adata.obs['pct_counts_ribo'] < 50, :]
    adata.raw = sc.pp.log1p(adata, copy=True)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    



def read_fasta(fasta_file):
    """ fasta to dict """
    dct_fasta = {}
    try:
        with open(fasta_file) as f:
            for l1 in f:
                if l1[0] != ">":
                    raise StopIteration
                l2 = f.readline()
                dct_fasta[l1[1:]] = l2
    except StopIteration:
        print("Error: is the file in the FASTA format ?")
    return dct_fasta


def read_fastq(fasta_file):
    """ fasta to dict """
    dct_fasta = {}
    try:
        with open(fasta_file) as f:
            for l1 in f:
                if l1[0] != "@":
                    raise StopIteration
                l2 = f.readline()
                f.readline()
                f.readline()
                dct_fasta[l1[1:].strip()] = l2.strip()
    except StopIteration:
        print("Error: is the file in the FASTA format ?")
    return dct_fasta


def to_fasta(df, fname, ind, sequence):
    """ df the dataframe containing the index values (column ind)
    and the sequences (column sequence). Saved in file fname
    """
    indexes = df[ind].values
    sequences = df[sequence].values
    with open(fname, "w") as f:
        for i, s in zip(indexes, sequences):
            f.write(">" + i + "\n")
            f.write(s + "\n")


def levenshtein_neighbours(d1, d2, threshold=1):
    """ find the pairs of sequences in d1 x d2 which have a Levenshtein distance
    lower than threshold """
    p = []
    for a1 in d1:
        for a2 in d2:
            if(a1 <= a2 and abs(len(a1) - len(a2)) <= threshold):
                if levenshtein.eval(a1, a2) <= threshold:
                    p.append((a1, a2))
    return p


def separate_chains(filename):
    """ Use mixcr to determine if a sequence is TRA or TRB, then create two files
    filename_tra et filename_trb
    """
    df = mixcr(filename)
    split_nm = filename.split(".")
    to_fasta(df[df.Chains == "TRA"],
             ".".join(split_nm[:-1])+"_tra."+split_nm[-1],
             "descrsR1", "targetSequences")
    to_fasta(df[df.Chains == "TRB"],
             ".".join(split_nm[:-1]) + "_trb." + split_nm[-1],
             "descrsR1", "targetSequences")
    return df


def cluster_heatmap(df, **kwargs):
    """ Given a numeric dataframe df with k rows and l columns,
    plot a heatmap k x l, and a dendrogramme
    
    """
    mat = df.values
    mat = np.exp(-mat)
    mat = mat + np.transpose(mat) - 2*np.diag(np.diag(mat))
    linkage = scipy.cluster.hierarchy.linkage(
        scipy.spatial.distance.squareform(mat), method='average')
    sns.clustermap(df, row_linkage=linkage, col_linkage=linkage, **kwargs)


import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib import colors

import warnings
import seaborn as sns
import math
import numpy as np
import pandas as pd


def get_idx_interv(d, D):
    k = 0
    while(d > D[k]):
        k += 1
    return k-1


def dist(A, B):
    return np.linalg.norm(np.array(A) - np.array(B))


def deCasteljau(b, t):
    N = len(b)
    a = np.copy(b)
    for r in range(1, N):
        a[:N-r, :] = (1-t)*a[:N-r, :] + t*a[1:N-r+1, :]
    return a[0, :]


def BezierCv(b, nr=5):
    t = np.linspace(0, 1, nr)
    return np.array([[deCasteljau(b, t[k]),
                      deCasteljau(b, t[k+1])] for k in range(nr-1)])


def position_circle(x, radius=1):
    """ Return the x,y coordinate of the point at
        angle (360*x)°, in the circle of radius "radius"
        and center (0, 0)
    """
    return np.array([radius*math.cos(x*2*math.pi),
                     radius*math.sin(x*2*math.pi)])


def linear_gradient(start, end, n=10):
    """ Return a gradient between start and end,
        with n elements.
    """
    gradients = np.zeros((len(start), n))
    for i in range(len(start)):
        gradients[i, :] = np.linspace(start[i], end[i], num=n)
    return np.transpose(gradients)


def linear_gradient_color(c1, c2, n=10):
    """ Return a gradient between the two color c1 & c2
    """
    return linear_gradient(colors.to_rgba(c1), colors.to_rgba(c2), n=n)


def draw_chord(A, B, ax=None, color_start="b", color_end="r",
               bezier_prms=([0, 0.765, 1.414, 1.848, 2], [1.2, 1.5, 1.8, 2.]),
               precision=1000,
               **kwargs):
    """ Draw a Bezier curve between two points """
    d = dist(A, B)

    # Depending on the distance between the points
    # the Bezier curve parameters change
    K = get_idx_interv(d, bezier_prms[0])
    b = [A, A/(1 + d),#bezier_prms[1][K],
         B/(1 + d),#bezier_prms[1][K],
         B]
    bz = BezierCv(b, nr=100)

    lc = mcoll.LineCollection(bz,
                              colors=linear_gradient_color(color_start,
                                                           color_end,
                                                           n=100), **kwargs)
    ax.add_collection(lc)


def draw_arc_circle(start, end, color="b", radius=1, ax=None,
                    thickness=0.1, precision=1000):
    ts = np.linspace(start, end, precision)
    poly_nodes = ([position_circle(t, radius=radius) for t in ts] +
                  [position_circle(t, radius=radius+thickness) for t in ts[::-1]])
    x, y = zip(*poly_nodes)
    ax.fill(x, y, color=color)


def add_text_circle(x, txt, radius=1, ax=None, **kwargs):
    ax.text(*position_circle(x, radius=radius),
            txt, rotation=360*x, ha='center', va='center', **kwargs)



def order_data(nodes, links, order=None, additional_cat={}):
    """
    Return a correctly ordered dataframe, ready to be plotted
    @ Args:
    - nodes: a dictionary that associates to each unique nodes a category name
    - links: link nodes together
    - order: order of the categories
    - additional_cat: a dict of dict a dict that associates to each unique node a name
    """
    categories = list(set(nodes.values()))
    ndsnb = dict(zip(nodes.keys(), range(len(nodes))))
    ## decide on the order
    if(order is None):
        order = sorted(categories)
    df = pd.Series(nodes).to_frame("categorie")
    df["nbcat"] = df.categorie.map(dict(zip(order, range(len(order)))))

    df = df.rename_axis('source').reset_index()
    df["nbsource"] = df.source.map(ndsnb)

    for ii in additional_cat:
        df[ii] = df.source.map(additional_cat[ii])
    
    
    dict_links = {}
    for l1, l2 in links:
        dict_links[l1] = l2
        dict_links[l2] = l1
        
    df["target"] = df.source.map(dict_links)
    df["nbtarget"] = df.target.map(ndsnb)
    df["tgt_nbcat"] = df.target.map(df.set_index("source").nbcat)

    
    df["tgt_cat_order"] = df.apply(lambda r: (len(order) - 1 + r["nbcat"] - r["tgt_nbcat"])%(len(order)-1), axis=1)
    df["sort_order"] = df.apply(lambda r: (r["nbsource"] if r["nbcat"] <= r["tgt_nbcat"] else -r["nbtarget"]), axis=1)
    df = df.sort_values(by=["nbcat", "tgt_cat_order", "sort_order"])
    return df
   

def chord_diagram(source, target, hue="black", data=None, ax=None, hue_order=None,
           palette=sns.color_palette(), categorie_internal_chords=False, sub_circle=False, spacing=0, spacing_sub=0, inverted=False,
                  precision_chord=100, precision_circle=100, thickness_circle=0.1, thickness_sub_circle=0.05, no_chords=False,
                  radius_text=1.1, threshold=0., circle_args={}, text_args={}, chord_args={}, 
                  additional_circle=None, spacing_add=0, thickness_add_circle=0.05, palette_add_circle=None):
    """ Draw a chord diagram
    @ Arguments:
    - source: A unique index for each of the chords. The order
    of these index will determine the order of the chord on the
    circle (starting at (1,0)). If data is not None, can be a column
    name, else should be an array.
    - target: The index to which the source are attached. If data is
    not None can be a column name, else should be an array.
    - hue: Can be either a color name, a column name (if data is not
    None), or an array. In the two last case, the color are given 
    by palette in order.
    - hue_order: The order in which the hue should be drawn
    - data: None or a dataframe containing columns source and target
    - ax: matplotlib.ax object in which to draw. If None, a figure 
    is created.
    - palette: the color palette used, if categorical data (default,
    seaborn default)
    - precision_chord: roughly the number of dot used to draw the chord
    bigger = slower but prettier
    - precision_circle: roughly the number of dot used to draw the circle
    bigger = slower but prettier
    - thickness_circle: thickness of the circle at the boundary
    - no_chords: if true don't draw the chords
    - radius_text: distance between the text and the center of the circle
    - categorie_internal_chords: if False does not draw a chord that
    start and end in the same categorie.
    - circle_args: argument of the ax.fill matplotlib function that
    draws the border of the circle.
    - text_args: argument of the ax.text matplotlib function that 
    draws the text.
    - chords_args: argument of the LineCollection function that
    draws the chords of the diagram.
    """
    if data is not None:
        source = data[source].values
        target = data[target].values
        if hue in data.keys():
            hue = data[hue].values

    if hue_order is None:
        hue_order = list(dict.fromkeys(hue)) 
    if additional_circle is not None:
        addit_data = dict(zip(source, data[additional_circle]))
    nodes = dict(zip(source, hue))

    links = list(zip(source, target))
    

    df = order_data(nodes, links, hue_order, 
                    additional_cat={} if additional_circle is None else {additional_circle: addit_data})
    
    idxs = list(np.where(df.nbcat.values[:-1] != df.nbcat.values[1:])[0])
    x = 0
    positions = []
    for i in range(len(df)):
        positions.append(x)
        if i in idxs:
            x += spacing
        x += (1 - spacing*(len(idxs)+1))/(len(df))
    df["position"] = positions
    df["tgt_position"] = df.target.map(df.set_index("source").position)
        
    if(len(palette) < len(hue_order)):
        warnings.warn("Not enough colors in the palette ({} needed), switching "
                      "to Seaborn husl palette.".format(len(hue_order)))
        palette = sns.color_palette("husl", len(order))

    if(ax is None):
        fig, ax = plt.subplots()

    nb_to_name_cat = dict(enumerate(hue_order))
    positions = df.position.values
    tgt_cat = df.nbcat.values
    idxs = np.where(tgt_cat[:-1] != tgt_cat[1:])[0]
    start_categorie = [0] + list(positions[idxs+1])
    end_categorie = list(positions[idxs]) + [positions[-1]]
    cats = [tgt_cat[0]] + list(tgt_cat[idxs+1])
    
    
    for s, e, c in zip(start_categorie, end_categorie, cats):
        draw_arc_circle(s - 0.5/len(df), e + 0.5/len(df), color=palette[c], ax=ax,
                        precision=precision_circle, thickness=thickness_circle,
                        radius=(1+thickness_sub_circle + spacing_sub*2*math.pi if inverted else 1),**circle_args)
        add_text_circle((s + e - 1/len(df))/2, nb_to_name_cat[c], ax=ax, color=palette[c], radius=radius_text, **text_args)

    if sub_circle:
        df["both_cat"] = df.apply(lambda r: str(r["nbcat"]) + "_" + str(r["tgt_nbcat"]), axis=1)
        positions = df.position.values
        tgt_cat = df.both_cat.values
        idxs = np.where(tgt_cat[:-1] != tgt_cat[1:])[0]
        start_categorie = [0] + list(positions[idxs+1])
        end_categorie = list(positions[idxs]) + [positions[-1]]
        subcat = [df.tgt_nbcat.values[0]] + list(df.tgt_nbcat.values[idxs+1])
        for s, e, c in zip(start_categorie, end_categorie, subcat):
            draw_arc_circle(s-0.5/len(df), e + 0.5/len(df), color=palette[c], ax=ax, precision=precision_circle,
                            thickness=thickness_sub_circle,
                            radius=(1 if inverted else 1+thickness_circle+spacing_sub*2*math.pi), **circle_args)
            
    if additional_circle is not None:
        df["circle_cat"] = df.apply(lambda r: str(r["nbcat"]) + "_" + str(r[additional_circle]), axis=1)
        positions = df.position.values
        circle_cat = df.circle_cat.values
        idxs = np.where(circle_cat[:-1] != circle_cat[1:])[0]
        start_categorie = [0] + list(positions[idxs+1])
        end_categorie = list(positions[idxs]) + [positions[-1]]
        subcat = [df[additional_circle].values[0]] + list(df[additional_circle].values[idxs+1])
        for s, e, c in zip(start_categorie, end_categorie, subcat):
            draw_arc_circle(s-0.5/len(df), e + 0.5/len(df), color=palette_add_circle[c], ax=ax, precision=precision_circle,
                            thickness=thickness_add_circle,
                            radius=(1 if inverted else 1+thickness_circle+spacing_add*2*math.pi), **circle_args)
        
    from collections import defaultdict
    percent = defaultdict(float)
    tot = defaultdict(float)
    totsrc = defaultdict(float)
    for src_c, tgt_c in zip(df.nbcat.values, df.tgt_nbcat.values):
        tot[tgt_c, src_c] += 1
        totsrc[src_c] += 1
    for src_c, tgt_c in tot:
        percent[tgt_c, src_c] = tot[tgt_c, src_c]/totsrc[src_c] 
    
        
    for jj, src_p, tgt_p, src_c, tgt_c in zip(range(len(df)), df["position"].values, 
                                          df["tgt_position"].values,
                                          df["nbcat"].values, 
                                          df["tgt_nbcat"].values):
        if (not (no_chords or (src_p == tgt_p or (not categorie_internal_chords and src_c == tgt_c)))
               and percent[src_c, tgt_c] > threshold):
            draw_chord(position_circle(src_p),
                        position_circle(tgt_p), ax=ax,
                        color_start=palette[(tgt_c if inverted else src_c)],
                        color_end=palette[(src_c if inverted else tgt_c)],
                        precision=precision_chord,
                        **chord_args)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('equal')
    ax.axis('off')
    return df



def order_data_section(nodes, links, order=None, sections=None, order_sections=None):
    """
    Return a correctly ordered dataframe, ready to be plotted
    @ Args:
    - nodes: a dictionary that associates to each unique nodes a category name
    - links: link nodes together
    - order: order of the categories
    - subsections: None or a dictionary that associates to each unique nodes a subsection
    """
    categories = list(set(nodes.values()))
    ndsnb = dict(zip(nodes.keys(), range(len(nodes))))
    ## decide on the order
    if(order is None):
        order = sorted(categories)
    if(order_sections is None):
        order_sections = sorted(list(set(sections.values())))
    df = pd.Series(nodes).to_frame("categorie")
    df["nbcat"] = df.categorie.map(dict(zip(order, range(len(order)))))

    df = df.rename_axis('source').reset_index()
    df["section"] = df.source.map(sections)
    
    unique_sections = list(set(sections.values()))
    dct_nb_section = dict(zip(unique_sections, range(len(unique_sections))))

    df["nbsection"] = df.section.map(dct_nb_section)
    df["nbsource"] = df.source.map(ndsnb)
    
    dict_links = {}
    for l1, l2 in links:
        dict_links[l1] = l2
        dict_links[l2] = l1
        
    df["target"] = df.source.map(dict_links)
    df["nbtarget"] = df.target.map(ndsnb)
    df["tgt_categorie"] = df.target.map(df.set_index("source").categorie)
    df["tgt_nbcat"] = df.target.map(df.set_index("source").nbcat)
    df["tgt_nbsection"] = df.target.map(df.set_index("source").nbsection)
    df["tgt_section"] = df.target.map(df.set_index("source").section)

    to_order = dict()
    n = 0
    for b in order_sections:
        for a in order:
            to_order[b, a] = n
            n += 1
    
    df["nb_sub"] = [to_order[b, a] for b, a in zip(df.section, df.categorie)]
    df["tgt_nb_sub"] = [to_order[b, a] for b, a in zip(df.tgt_section, df.tgt_categorie)]
    
    df["sort_order"] = df.apply(lambda r: (r["nbsource"] if r["nb_sub"] <= r["tgt_nb_sub"] else -r["nbtarget"]), axis=1)
    df = df.sort_values(by=["nb_sub",#"tgt_nbsection_order",
                            "tgt_nb_sub",
                            "sort_order"])
    return df
   
    
def chord_diagram_section(source, target, hue="black", section=None, data=None, ax=None, hue_order=None,
                  order_section=None,
           palette=sns.color_palette(), categorie_internal_chords=False, 
                  sub_circle=False, spacing=0, spacing_section=0,
                  spacing_between_circles=0, inverted=False,
                  precision_chord=100, precision_circle=100, thickness_circle=0.1,
                  thickness_sub_circle=0.05, no_chords=False,
                  radius_text=1.1, radius_text_section=1.3, circle_args={}, text_args={}, chord_args={},
                 section_text_args={}, hatch_section=False, list_hatches=None):
    """ Draw a chord diagram
    @ Arguments:
    - source: A unique index for each of the chords. The order
    of these index will determine the order of the chord on the
    circle (starting at (1,0)). If data is not None, can be a column
    name, else should be an array.
    - target: The index to which the source are attached. If data is
    not None can be a column name, else should be an array.
    - hue: Can be either a color name, a column name (if data is not
    None), or an array. In the two last case, the color are given 
    by palette in order.
    - section: If not None, will cut the chord diagram in sections.
    Can be a column name (if data is not None) or an array.
    - hue_order: The order in which the hue should be drawn
    - data: None or a dataframe containing columns source and target
    - ax: matplotlib.ax object in which to draw. If None, a figure 
    is created.
    - palette: the color palette used, if categorical data (default,
    seaborn default)
    - precision_chord: roughly the number of dot used to draw the chord
    bigger = slower but prettier
    - precision_circle: roughly the number of dot used to draw the circle
    bigger = slower but prettier
    - thickness_circle: thickness of the circle at the boundary
    - no_chords: if true don't draw the chords
    - radius_text: distance between the text and the center of the circle
    - categorie_internal_chords: if False does not draw a chord that
    start and end in the same categorie.
    - circle_args: argument of the ax.fill matplotlib function that
    draws the border of the circle.
    - text_args: argument of the ax.text matplotlib function that 
    draws the text.
    - chords_args: argument of the LineCollection function that
    draws the chords of the diagram.
    """
    if data is not None:
        source = data[source].values
        target = data[target].values
        if hue in data.keys():
            hue = data[hue].values
        if section is not None:
            section = data[section].values
            
    if hue_order is None:
        hue_order = list(dict.fromkeys(hue))
        
    
        
    nodes = dict(zip(source, hue))

    links = list(zip(source, target))
    
    df = order_data_section(nodes, links, hue_order,
                    (None if section is None else dict(zip(source, section))), order_section)

    idxs = list(np.where(df.nbcat.values[:-1] != df.nbcat.values[1:])[0])
    idxs_sections = list(np.where(df.nbsection.values[:-1] != df.nbsection.values[1:])[0])
    x = 0
    positions = []
    for i in range(len(df)):
        positions.append(x)
        if i in idxs:
            x += spacing
        if i in idxs_sections:
            x += spacing_section - spacing
            
        x += (1 - spacing*(len(idxs)+1) 
              - (spacing_section - spacing)*(len(idxs_sections)+1))/(len(df))
    df["position"] = positions
    df["tgt_position"] = df.target.map(df.set_index("source").position)
        
    if(len(palette) < len(hue_order)):
        warnings.warn("Not enough colors in the palette ({} needed), switching "
                      "to Seaborn husl palette.".format(len(hue_order)))
        palette = sns.color_palette("husl", len(order))

    if(ax is None):
        fig, ax = plt.subplots()

    nb_to_name_cat = dict(enumerate(hue_order))
    positions = df.position.values
    tgt_cat = df.nbcat.values
    idxs = np.where(tgt_cat[:-1] != tgt_cat[1:])[0]
    start_categorie = [0] + list(positions[idxs+1])
    end_categorie = list(positions[idxs]) + [positions[-1]]
    cats = [tgt_cat[0]] + list(tgt_cat[idxs+1])
    
    for s, e, c in zip(start_categorie, end_categorie, cats):
        draw_arc_circle(s - 0.5/len(df), e + 0.5/len(df), color=palette[c], ax=ax,
                        precision=precision_circle, thickness=thickness_circle,
                        radius=(1+thickness_sub_circle + spacing_between_circles*2*math.pi if inverted else 1),**circle_args)
        add_text_circle((s + e - 1/len(df))/2, nb_to_name_cat[c], ax=ax, color=palette[c], radius=radius_text, **text_args)


    if(section is not None):
        idxs_section = np.where(df.nbsection.values[:-1] != df.nbsection.values[1:])[0]
        start_sect = [0] + list(positions[idxs_section+1])
        end_sect = list(positions[idxs_section]) + [positions[-1]]
        section = [df.section.values[0]] + list(df.section.values[idxs_section+1])

        for s, e, c in zip(start_sect, end_sect, section):
            add_text_circle((s+e-1/len(df))/2, c, ax=ax, color="k", radius=radius_text_section, **section_text_args)

        
    if sub_circle:
        df["both_cat"] = df.apply(lambda r: str(r["nbcat"]) + "_" + str(r["tgt_nbcat"]), axis=1)
        positions = df.position.values
        tgt_cat = df.both_cat.values
        idxs = np.where(tgt_cat[:-1] != tgt_cat[1:])[0]
        start_categorie = [0] + list(positions[idxs+1])
        end_categorie = list(positions[idxs]) + [positions[-1]]
        subcat = [df.tgt_nbcat.values[0]] + list(df.tgt_nbcat.values[idxs+1])
        subsect = [df.tgt_nbsection.values[0]] + list(df.tgt_nbsection.values[idxs+1])
        
        if list_hatches is None:
            list_hatches = ['+', 'o', '/', '\\', '|', '-',
                           'x', 'O',
                          '.', '*']
        
        for s, e, c, st in zip(start_categorie, end_categorie, subcat, subsect):
            draw_arc_circle(s-0.5/len(df), e + 0.5/len(df), color=palette[c], ax=ax, precision=precision_circle,
                            thickness=thickness_sub_circle,
                            radius=(1 if inverted else 1+thickness_circle+spacing_between_circles*2*math.pi),
                            hatch=(None if not hatch_section else list_hatches[st%len(list_hatches)]),
                            **circle_args)

        
    for jj, src_p, tgt_p, src_c, tgt_c in zip(range(len(df)), df["position"].values, 
                                          df["tgt_position"].values,
                                          df["nbcat"].values, 
                                          df["tgt_nbcat"].values):
        if not (no_chords or (src_p == tgt_p or (not categorie_internal_chords and src_c == tgt_c))):
            draw_chord(position_circle(src_p),
                        position_circle(tgt_p), ax=ax,
                        color_start=palette[(int(tgt_c) if inverted else int(src_c))],
                        color_end=palette[(int(src_c) if inverted else int(tgt_c))],
                        precision=precision_chord,
                        **chord_args)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('equal')
    ax.axis('off')
    return df

    
# if __name__ == '__main__':
    # names = []
    # if len(sys.argv) > 1:
    #     names = [str(a) for a in sys.argv[1:]]

    # for name in names:
        # run_mixcr_tcr("../Data/" + name + "_VDJ/", filtered=True)
        # separate_chains("../Data/" + name + "_VDJ/outs/all_contig.fasta")
        # igor("../Data/" + name + "_VDJ/outs/all_contig.fasta",
        #      {"alpha": "../Data/" + name + "_VDJ/outs/all_contig_tra.fasta",
        #       "beta": "../Data/" + name + "_VDJ/outs/all_contig_trb.fasta"},
        #      output="../Data/" + name + "_VDJ/outs/all_contig_igor.tsv",
        #      intermediate_repertory="../Data/" + name + "_VDJ/igor/")
