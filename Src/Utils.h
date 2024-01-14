#pragma once

template <typename T>
T alignUp(T value, T alignment) {
	if (alignment == 0) {
		// Avoid division by zero; you may want to handle this differently depending on your needs
		return value;
	}

	T remainder = value % alignment;

	if (remainder != 0) {
		return value + (alignment - remainder);
	}

	return value;
}


