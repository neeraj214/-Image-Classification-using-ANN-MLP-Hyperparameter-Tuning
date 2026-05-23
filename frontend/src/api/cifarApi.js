const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Handles fetch responses and throws an error if the response is not OK.
 */
const handleResponse = async (response) => {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }
  return response.json();
};

/**
 * Uploads an image file to the backend for prediction.
 * @param {File} file - The image file to classify.
 * @returns {Promise<Object>} - The prediction results from CNN and MLP models.
 */
export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse(response);
};

/**
 * Fetches the benchmark comparison results from the backend.
 * @returns {Promise<Object>} - Consolidated benchmark metrics.
 */
export const getBenchmark = async () => {
  const response = await fetch(`${API_BASE_URL}/benchmark`);
  return handleResponse(response);
};

/**
 * Checks the health status of the backend API and loaded models.
 * @returns {Promise<Object>} - Status and list of loaded models.
 */
export const checkHealth = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(response);
};
