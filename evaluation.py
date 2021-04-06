def metrics(tp, fp, tn, fn, qid):
    if (tp+fp > 0):
        prec = tp/(tp+fp)
    else:	 	
        prec = 0.0
    if (tp+fn > 0):
        rec = tp/(tp+fn)
    else:		
        rec = 0.0
    if (prec + rec) != 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    accuracy = (tp+tn) / (tp + fp + tn + fn)
    if qid == 8:
        accuracy = f1   # this is because Q8 can have multiple valid answers and F1 makes more sense here
    total = tp + fp + tn + fn

    header = '\t'.join(["Total", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"])
    results = [total, tp, fp, tn, fn, accuracy*100, prec*100, rec*100, f1*100]
    results_str = "%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f" % (total, tp, fp, tn, fn, accuracy*100, prec*100, rec*100, f1*100)
    return (header, results_str, results)

def check_creation(states):
    steps = []
    for step in range(1, len(states)):
        if states[step-1] == "-" and states[step] != "-":
            steps.append(step)
    check = 0
    if len(steps) > 0:
        check = 1
    return steps, check 

def check_destroy(states):
    steps = []
    for step in range(1, len(states)):
        if states[step-1] != "-" and states[step] == "-":
            steps.append(step)
    check = 0
    if len(steps) > 0:
        check = 1
    return steps, check 

def check_move(states):
    steps = []
    for step in range(1, len(states)):
        if states[step-1] != "-" and states[step] != "-":
            if states[step-1] != states[step]:
                steps.append(step)
    check = 0
    if len(steps) > 0:
        check = 1
    return steps, check 

def Q1(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        # find predictions
        be_created = {}
        lab_created_participants = []
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            pred_creation_step, be_created[participant] = check_creation(predictions[pid]['entities'][participant])
            if check_creation(label['states'][participant_id])[1]:
                lab_created_participants.append(participant)
            tp += int(be_created[participant] and (participant in lab_created_participants))
            fp += int(be_created[participant] and (participant not in lab_created_participants))
            tn += int(not be_created[participant] and (participant not in lab_created_participants))
            fn += int(not be_created[participant] and (participant in lab_created_participants))
    return tp,fp,tn,fn

def Q2(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            pred_creation_step, _ = check_creation(predictions[pid]['entities'][participant])
            gold = check_creation(label['states'][participant_id])
            if gold[1]:
                for pred in pred_creation_step:
                    tp += int(pred in gold[0])
                    fp += int(pred not in gold[0])
                for golden in gold[0]:
                    fn += int(golden not in pred_creation_step)
    return tp,fp,tn,fn

def Q3(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        # find predictions
        be_created = {}
        lab_created_participants = []
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            gold = check_creation(label['states'][participant_id])
            if gold[1]:
                pred_loc = predictions[pid]['entities'][participant][gold[0][0]]
                correct_loc = label['states'][participant_id][gold[0][0]]
                tp += int(pred_loc != "-" and pred_loc != "?" and pred_loc in correct_loc)
                fp += int(pred_loc != "-" and pred_loc != "?" and pred_loc not in correct_loc)
                fn += int(pred_loc == "-" or pred_loc == "?")
    return tp,fp,tn,fn

def Q4(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        # find predictions
        be_created = {}
        lab_created_participants = []
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            pred_creation_step, be_created[participant] = check_destroy(predictions[pid]['entities'][participant])
            if check_destroy(label['states'][participant_id])[1]:
                lab_created_participants.append(participant)
            tp += int(be_created[participant] and (participant in lab_created_participants))
            fp += int(be_created[participant] and (participant not in lab_created_participants))
            tn += int(not be_created[participant] and (participant not in lab_created_participants))
            fn += int(not be_created[participant] and (participant in lab_created_participants))
    return tp,fp,tn,fn

def Q5(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            pred_creation_step, _ = check_destroy(predictions[pid]['entities'][participant])
            gold = check_destroy(label['states'][participant_id])
            if gold[1]:
                for pred in pred_creation_step:
                    tp += int(pred in gold[0])
                    fp += int(pred not in gold[0])
                for golden in gold[0]:
                    fn += int(golden not in pred_creation_step)
    return tp,fp,tn,fn

def Q6(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        # find predictions
        be_created = {}
        lab_created_participants = []
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            gold = check_destroy(label['states'][participant_id])
            if gold[1]:
                pred_loc = predictions[pid]['entities'][participant][gold[0][0]-1]
                correct_loc = label['states'][participant_id][gold[0][0]-1]
                tp += int(pred_loc != "-" and pred_loc != "?" and pred_loc in correct_loc)
                fp += int(pred_loc != "-" and pred_loc != "?" and pred_loc not in correct_loc)
                fn += int(pred_loc == "-" or pred_loc == "?")
    return tp,fp,tn,fn

def Q7(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        # find predictions
        be_created = {}
        lab_created_participants = []
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            pred_creation_step, be_created[participant] = check_move(predictions[pid]['entities'][participant])
            if check_move(label['states'][participant_id])[1]:
                lab_created_participants.append(participant)
            tp += int(be_created[participant] and (participant in lab_created_participants))
            fp += int(be_created[participant] and (participant not in lab_created_participants))
            tn += int(not be_created[participant] and (participant not in lab_created_participants))
            fn += int(not be_created[participant] and (participant in lab_created_participants))
    return tp,fp,tn,fn

def Q8(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            pred_creation_step, _ = check_move(predictions[pid]['entities'][participant])
            gold = check_move(label['states'][participant_id])
            if gold[1]:
                for pred in pred_creation_step:
                    tp += int(pred in gold[0])
                    fp += int(pred not in gold[0])
                for golden in gold[0]:
                    fn += int(golden not in pred_creation_step)
    return tp,fp,tn,fn

def Q9(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        # find predictions
        be_created = {}
        lab_created_participants = []
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            gold = check_move(label['states'][participant_id])
            if gold[1]:
                for golden in gold[0]:
                    pred_loc = predictions[pid]['entities'][participant][golden-1]
                    correct_loc = label['states'][participant_id][golden-1]
                    tp += int(pred_loc != "-" and pred_loc != "?" and pred_loc in correct_loc)
                    fp += int(pred_loc != "-" and pred_loc != "?" and pred_loc not in correct_loc)
                    fn += int(pred_loc == "-" or pred_loc == "?")
    return tp,fp,tn,fn

def Q10(labels, predictions):
    tp = fp = tn = fn = 0.0
    for label in labels:
        pid = label['para_id']
        setParticipants = label['participants']
        # find predictions
        be_created = {}
        lab_created_participants = []
        for participant_id in range(len(setParticipants)):
            participant = setParticipants[participant_id]
            gold = check_move(label['states'][participant_id])
            if gold[1]:
                for golden in gold[0]:
                    pred_loc = predictions[pid]['entities'][participant][golden]
                    correct_loc = label['states'][participant_id][golden]
                    tp += int(pred_loc != "-" and pred_loc != "?" and pred_loc in correct_loc)
                    fp += int(pred_loc != "-" and pred_loc != "?" and pred_loc not in correct_loc)
                    fn += int(pred_loc == "-" or pred_loc == "?")
    return tp,fp,tn,fn

def get_metrics(labels, predictions):
    scores = {}
    tp, fp, tn, fn = Q1(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,1)
    scores[1] = results[5]

    tp, fp, tn, fn = Q4(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,4)
    scores[4] = results[5]

    tp, fp, tn, fn = Q7(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,7)
    scores[7] = results[5]

    cat1_score = (scores[1] + scores[4] + scores[7]) / 3
    print("cat1 is : ", cat1_score)
    
    tp, fp, tn, fn = Q2(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,2)
    scores[2] = results[5]

    tp, fp, tn, fn = Q5(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,5)
    scores[5] = results[5]

    tp, fp, tn, fn = Q8(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,8)
    scores[8] = results[5]

    cat2_score = (scores[2] + scores[5] + scores[8]) / 3
    print("cat2 is : ", cat2_score)
    
    tp, fp, tn, fn = Q3(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,3)
    scores[3] = results[5]

    tp, fp, tn, fn = Q6(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,6)
    scores[6] = results[5]

    tp, fp, tn, fn = Q9(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,9)
    scores[9] = results[5]

    tp, fp, tn, fn = Q10(labels, predictions)
    header,results_str, results = metrics(tp,fp,tn,fn,10)
    scores[10] = results[5]

    cat3_score = (scores[3] + scores[6] + scores[9] + scores[10]) / 4
    print("cat3 is : ", cat3_score)