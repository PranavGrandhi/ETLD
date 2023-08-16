import numpy as np

standard_aa = ['ala', 'cys', 'asp', 'glu', 'phe', 'gly',
       'his', 'ile', 'lys', 'leu', 'met', 'asn',
       'pro', 'gln', 'arg', 'ser', 'thr', 'val',
       'trp', 'tyr']


def max_res_id_from_pdb(pdbfile):
    res_index = []
    fr = open(pdbfile, 'r')
    for row in fr:
        if row.startswith('ATOM'):
            index = int(float(row[22:26]))
            res_index.append(index)
    fr.close()
    return max(res_index)

def seq_len_from_msa(msafile, aa_isupper=False):

    have_key = False
    if open(msafile).read()[0] == '>':
        have_key = True

    fr = open(msafile, 'r')
    seq = ''
    if have_key:
        for row in fr.readlines()[1:]:
            if row.startswith('>'):
               break
            else:
                seq += row.strip()
    else:
        seq = fr.readline().strip()
    fr.close()

    if aa_isupper:
        seq = [s for i, s in enumerate(list(seq)) if s.isupper()]

    return len(seq)

def rrdist_from_pdb(pdbfile, min_res_delta=5, mask_threshold=8):
    coords = []

    fr = open(pdbfile, 'r')
    for row in fr:
        if row.startswith('ATOM') or row.startswith('HETATM'):
            atomName = row[12:16].strip()
            resName = row[17:20].strip().lower()
            if resName in standard_aa:
                if resName == 'gly' and atomName == 'CA':
                    x, y, z = float(row[31:38]), float(row[38:46]), float(row[46:54])
                    coords.append(np.array([x, y, z]))
                elif atomName == 'CB':
                    x, y, z = float(row[31:38]), float(row[38:46]), float(row[46:54])
                    coords.append(np.array([x, y, z]))
    fr.close()

    atomnum = len(coords)
    distmatrix = np.zeros((atomnum, atomnum))
    mask = np.zeros((atomnum, atomnum))
    for n in range(atomnum - min_res_delta):
        for m in range(n + min_res_delta, atomnum):
            dist = np.sqrt(np.sum((coords[n] - coords[m]) ** 2))
            # dist = np.log(dist)
            distmatrix[n, m] = dist
            distmatrix[m, n] = dist
            if dist <= mask_threshold:
                mask[n, m] = 1
                mask[m, n] = 1
    np.seterr(divide='ignore', invalid='ignore')
    distmatrix = 1. / distmatrix
    distmatrix[np.isnan(distmatrix)] = 0
    return distmatrix, mask

def mask_from_rrdist(distfile, min_res_delta=5, mask_threshold=8):
    if type(distfile) is str:
        try:
            dist = np.loadtxt(distfile)
        except:
            dist = np.load(distfile)
    else:
        dist = distfile
    atomnum = dist.shape[0]
    mask = np.zeros((atomnum, atomnum))
    for n in range(atomnum - min_res_delta):
        for m in range(n + min_res_delta, atomnum):
            if dist[n, m] <= mask_threshold:
                mask[n, m] = 1
                mask[m, n] = 1
    np.seterr(divide='ignore', invalid='ignore')
    return mask


