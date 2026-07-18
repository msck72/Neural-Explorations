import { useState } from "react"
import defaultImage from "../assets/elephant.jpeg"

const VITE_FILTER_SERVICE_API = import.meta.env.VITE_FILTER_SERVICE_API;

function ApplyFilters() {

    const [image, setImage] = useState(defaultImage);

    const [filterImages, setFilterImages] = useState({
            Blur: defaultImage,
            EdgeDetection: defaultImage,
            GaussianBlur: defaultImage,
            Sharpen: defaultImage,
            SobelX: defaultImage,
            SobelY: defaultImage
        });

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleImageChange = (e) => {
        const file = e.target.files[0];

        if(!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);
        setError("");
        setLoading(true);
        
        const applyInferenceEngineFilters = async () => {
            if (!VITE_FILTER_SERVICE_API) {
                setError('Filter service URL is not configured. Please set VITE_FILTER_SERVICE_API.');
                setLoading(false);
                return;
            }

            const formData = new FormData();
            formData.append("image", file);

            try{
                const response = await fetch(VITE_FILTER_SERVICE_API, {
                    method: "POST",
                    body: formData,
                    credentials: 'omit'
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText || 'Filter service returned an error');
                }

                const result = await response.json();
                const processed_images = Object.fromEntries(
                    Object.entries(result).map(([key, value]) => [
                        key,
                        `data:image/png;base64,${value}`,
                    ])
                );
                setFilterImages(prev => ({
                    ...prev,
                    ...processed_images,
                }));
            } catch (error) {
                setError('FILTER SERVICE FAILURE');
                console.log("Error occured during API call to FILTER_SERVICE: ", error);
            } finally {
                setLoading(false);
            }
        }

        applyInferenceEngineFilters();

    }

    function FiltersDivs() {
        return (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(filterImages).map(([filter, img]) => (
                    <div key={filter} className="bg-white shadow-sm rounded-lg overflow-hidden hover:shadow-lg transition-transform transform hover:-translate-y-1">
                        <div className="w-full h-40 flex items-center justify-center bg-gray-100">
                            <img
                                src={img}
                                alt={filter}
                                className="max-w-full max-h-full object-contain"
                            />
                        </div>
                        <div className="p-2 text-sm font-medium text-center bg-gray-50">{filter}</div>
                    </div>
                ))}
            </div>
        )
    }

    return (
        <div className="p-6">
            <div className="mx-auto bg-white/60 backdrop-blur-md rounded-2xl shadow-xl p-6">
                {/* <h2 className="text-2xl font-semibold mb-4">Apply Filters</h2> */}
                <div className="flex flex-col md:flex-row gap-6">
                    <div className="flex-1">
                        <div className="bg-linear-to-br from-white to-gray-50 rounded-lg p-4 shadow-inner border border-gray-200">
                            <div className="relative w-full flex items-center justify-center bg-gray-100 rounded-md border overflow-hidden">
                                <img
                                    src={image}
                                    alt="Preview"
                                    className="max-w-full max-h-full object-contain"
                                />
                                {loading && (
                                    <div className="absolute inset-0 bg-black/30 flex items-center justify-center rounded-md">
                                        <div className="flex items-center gap-3">
                                            <div className="border-4 border-t-blue-500 border-gray-200 rounded-full animate-spin" />
                                            <div className="text-white font-medium">Processing...</div>
                                        </div>
                                    </div>
                                )}
                            </div>
                            <div className="mt-4 flex items-center justify-center gap-4">
                                <label htmlFor="image-upload" className="px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition">Upload Image</label>
                                <input id="image-upload" type="file" accept="image/*" className="hidden" onChange={handleImageChange}/>
                                {error && <div className="text-red-600 text-sm">{error}</div>}
                            </div>
                        </div>
                    </div>
                    <div className="flex-1">
                        <div className="p-4 bg-gray-50 rounded-lg">
                            <h3 className="text-lg font-medium mb-3">Filtered Outputs</h3>
                            <FiltersDivs />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ApplyFilters