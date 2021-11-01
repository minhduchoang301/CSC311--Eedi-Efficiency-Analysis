from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, metadata,theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    for num in range(len(data['user_id'])):
        i = data['user_id'][num]
        j = data['question_id'][num]
        weight = 0.5
        for subj in metadata["subject_id"][j]:
            weight += alpha[i][subj]
        weight /= len(metadata["subject_id"][j]) 
        log_lklihood += weight*data['is_correct'][num]*(theta[i] - beta[j]) - np.log(1+ np.exp(weight*(theta[i] - beta[j])))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return log_lklihood


def update_theta_beta(data, metadata,lr, theta, beta, alpha):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_student = len(theta)
    num_question = len(beta)
    theta_delta = np.zeros(num_student)
    beta_delta = np.zeros(num_question)
    for num in range(len(data['user_id'])):
        student_id = data['user_id'][num]
        question_id = data['question_id'][num]
        
        weight = 0.5
        for subj in metadata["subject_id"][question_id]:
            weight += alpha[student_id][subj]
        weight /= len(metadata["subject_id"][question_id])

        sigmoid_value = weight*(theta[student_id]-beta[question_id])
        theta_delta[student_id] += (weight*data['is_correct'][num] - weight*sigmoid(sigmoid_value))
    theta += lr * theta_delta
    for num in range(len(data['user_id'])):
        student_id = data['user_id'][num]
        question_id = data['question_id'][num]
        
        weight = 0.5
        for subj in metadata["subject_id"][question_id]:
            weight += alpha[student_id][subj]
        weight /= len(metadata["subject_id"][question_id])
        
        sigmoid_value = weight*(theta[student_id]-beta[question_id])
        beta_delta[question_id] += (-data['is_correct'][num]*weight + weight*sigmoid(sigmoid_value))
    beta += lr * beta_delta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, metadata,val_data, matrix, lr, iterations, alpha):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(matrix.shape[0])
    beta = np.zeros(matrix.shape[1])

    val_acc_lst = []
    train_nllk_lst = []
    val_nllk_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, metadata,theta=theta, beta=beta, alpha=alpha)
        neg_val_lld = neg_log_likelihood(val_data, metadata,theta=theta, beta=beta, alpha=alpha)
        train_nllk_lst.append(neg_lld)
        val_nllk_lst.append(neg_val_lld)
        score = evaluate(data=val_data, metadata = metadata,theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, metadata,lr, theta, beta, alpha=alpha)

    # TODO: You may change the return values to achieve what you want.
    return train_nllk_lst, val_nllk_lst, theta, beta, val_acc_lst


def evaluate(data, metadata,theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        
        weight = 0.5
        for subj in metadata["subject_id"][q]:
            weight += alpha[u][subj]
        weight /= len(metadata["subject_id"][q])
        
        x = (theta[u] - beta[q]).sum()*0.75+ weight*0.25
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def load_meta():
    path = os.path.join("data", "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
        
    data = {
        "question_id": [],
        "subject_id": []
    }
    
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                news = row[1].replace(" ","").replace("[","").replace("]", "").split(",")
                data["question_id"].append(int(row[0]))
                data["subject_id"].append(list(map(int, news)))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data

def getAlpha(metadata, train_data):
    precalc = dict()
    for num in range(len(train_data['user_id'])):
        student_id = train_data['user_id'][num]
        question_id = train_data['question_id'][num]
        if student_id not in precalc.keys():
            precalc[student_id] = dict()
            precalc[student_id]["count"] = np.ones(388)
            precalc[student_id]["sum"] = np.zeros(388)
        for subj_id in metadata["subject_id"][question_id]:
                precalc[student_id]["count"][subj_id] += 1
                precalc[student_id]["sum"][subj_id] += train_data["is_correct"][num]
    alpha = np.zeros((max(precalc.keys())+1, 388))
    for key in precalc.keys():
        alpha[key] = np.true_divide(precalc[key]["sum"], precalc[key]["count"])
    return alpha

def main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")
    metadata  = load_meta()
    alpha = getAlpha(metadata, train_data)
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rate = 0.015
    num_iterations = 75
    train_nllk_lst, val_nllk_lst, theta, beta, val_acc_lst = irt(train_data, metadata,val_data, sparse_matrix.toarray(), learning_rate, num_iterations, alpha)
    plt.plot(range(0, num_iterations), train_nllk_lst, label = "training log-likelihood")
    plt.plot(range(0, num_iterations), val_nllk_lst, label = "validating log-likelihood")
    plt.ylabel("negative log-likelihood")
    plt.xlabel("number of iterations")
    plt.xticks(range(0, num_iterations, 5))
    plt.legend()
    plt.show()
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    # we calculate this based on the number of iteration being 9, not 25
    # we calculate the validation_accuracy again since in the irt qfunction, we calculate the validation_accracy before updating theta and beta, and therefore we need to calculate again to find the validation accuracy of the final theta and beta value
    final_num_iteration = np.argmax(val_acc_lst)
    validation_accuracy = evaluate(val_data, metadata,theta, beta, alpha)
    testing_accuracy = evaluate(test_data, metadata,theta, beta, alpha)
    print("final validation accuracy:", validation_accuracy)
    print("final testing accuracy:", testing_accuracy)
    #####################################################################
    # part (d)
    # we calculate this part with the number of iterations being 500
    q = [0, 10, 100, 1000, 1773]
    for question in q:
        p = []
        for theta in range(-3, 4):
            p.append(sigmoid(theta - beta[question]))
        plt.plot(range(-3, 4), p, label = "question {}".format(question))
    plt.ylabel("proability of correct response p(cij = 1)")
    plt.xlabel("theta values")
    plt.xticks(range(-3, 4))
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
