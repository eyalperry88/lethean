import torch
import torch.nn as nn

from utils.train_helpers import *
from utils.rotation import rotate_batch, rotate_single_with_label

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trerr_single(model, image):
	model.eval()
	labels = torch.LongTensor([0, 1, 2, 3])
	inputs = []
	for label in labels:
		inputs.append(rotate_single_with_label(rotation_te_transforms(image), label))
	inputs = torch.stack(inputs)
	inputs, labels = inputs.cuda(), labels.cuda()
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
	return predicted.eq(labels).cpu()

def adapt_single(model, image, optimizer, criterion, niter, batch_size):
	model.train()
	for iteration in range(niter):
		inputs = [te_transforms(image) for _ in range(batch_size)]
		print("a")
		inputs, labels = rotate_batch(inputs)
		print("b")
		inputs, labels = inputs.to(device), labels.to(device)
		print("c")
		optimizer.zero_grad()
		print("d")
		_, ssh = model(inputs)
		print("e")
		loss = criterion(ssh, labels)
		print("f")
		loss.backward()
		print("g")
		optimizer.step()
		print("h")

def adapt_single_tensor(model, tensor, optimizer, criterion, niter, batch_size):
	model.train()
	for iteration in range(niter):
		inputs = [tensor for _ in range(batch_size)]
		inputs, labels = rotate_batch(inputs)
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		_, ssh = model(inputs)
		loss = criterion(ssh, labels)
		loss.backward()
		optimizer.step()


def test_single(model, image, label):
	model.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs, outputs_ssh = model(inputs.to(device))
		_, predicted = outputs.max(1)
		confidence = nn.functional.softmax(outputs_ssh, dim=1).squeeze()[0].item()
	correctness = 1 if predicted.item() == label else 0
	return correctness, confidence
