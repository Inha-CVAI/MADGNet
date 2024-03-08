import os

def save_metrics(args, test_results, model_dirs, current_epoch):
    print("###################### TEST REPORT ######################")
    for metric in test_results.keys():
        print("Mean {}    :\t {}".format(metric, test_results[metric]))
    print("###################### TEST REPORT ######################\n")

    if args.train_data_type == args.test_data_type:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {}).txt'.format(current_epoch))
    else:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'Generalizability test_reports({}->{}).txt'.format(args.train_data_type, args.test_data_type))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in test_results.keys():
        f.write("Mean {}    :\t {}\n".format(metric, test_results[metric]))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))