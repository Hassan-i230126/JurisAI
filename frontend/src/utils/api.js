export const getApiUrl = (endpoint) => {
  const host = import.meta.env.VITE_BACKEND_HOST;
  if (!host) return endpoint;
  
  const protocol = host.includes('localhost') ? 'http' : 'https';
  return \://System.Management.Automation.Internal.Host.InternalHost;
};
