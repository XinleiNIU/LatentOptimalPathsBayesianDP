def moving_average(loss,n=5):

	# Build a tensor list
	loss_list = []

	if len(loss_list) >= n:
		# drop the first loss
		loss_list =  loss_list[1:]

	loss_list.appned(loss.detach())

	return loss_list.sum()/n
