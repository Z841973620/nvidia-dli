def run_assessment(model, test_it):
    print('Evaluating model 5 times to obtain average accuracy...\n')
    averages = [model.evaluate(test_it, steps=test_it.samples/test_it.batch_size)[1] for i in range(5)]
    average = sum(averages) / len(averages)
    print('\nAccuracy required to pass the assessment is 0.92 or greater.')
    print('Your average accuracy is {:5.4f}.\n'.format(average))
    
    if average >= .92:
        print('Congratulations! You passed the assessment!')
    else:
        print('Your accuracy is not yet high enough to pass the assessment, please continue trying.')
