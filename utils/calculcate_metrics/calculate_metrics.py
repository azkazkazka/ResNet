from calculate_modules import *
import a_dcf

# input (array of cm-score, array of label)
# cm-score: probability or log likelihood ratio of bonafide prediction
# label: array of 0 (spoof) and 1 (bonafide)

def sigmoid_to_llr(scores):
    scores = np.clip(scores, 1e-10, 1 - 1e-10)
    return np.log(scores / (1 - scores))


def calculate_minDCF_EER_CLLR_actDCF(cm_scores, cm_keys):
    """
    Evaluation metrics for track 1
    Primary metrics: min DCF,
    Secondary metrics: EER, CLLR, actDCF
    """
    
    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }

    assert cm_keys.size == cm_scores.size, "Error, unequal length of cm label and score files"
    
    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 1]
    spoof_cm = cm_scores[cm_keys == 0]

    # print(bona_cm)
    # print(spoof_cm)

    # EERs of the standalone systems
    eer_cm, frr, far, thresholds, eer_threshold = compute_eer(bona_cm, spoof_cm)#[0]
    # cllr

    print(eer_threshold)
    accuracy = calculate_accuracy(cm_scores, cm_keys, threshold=eer_threshold)
    cmatrix = generate_confusion_matrix(cm_scores, cm_keys, threshold=eer_threshold)

    bona_cm = sigmoid_to_llr(np.array(bona_cm))
    spoof_cm = sigmoid_to_llr(np.array(spoof_cm))
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    # min DCF
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])
    # actual DCF
    actDCF, thres = compute_actDCF(bona_cm, spoof_cm, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    return minDCF_cm, eer_cm, cllr_cm, actDCF, accuracy, cmatrix