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
    
    const handleImageChange = (e) => {
        const file = e.target.files[0];

        if(!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);
        
        const applyInferenceEngineFilters = async () => {
            if (!VITE_FILTER_SERVICE_API) {
                alert('Filter service URL is not configured. Please set VITE_FILTER_SERVICE_API.');
                return;
            }

            const formData = new FormData();
            formData.append("image", file);

            try{
                console.log('Sending image to filter service', VITE_FILTER_SERVICE_API)
                const response = await fetch(VITE_FILTER_SERVICE_API, {
                    method: "POST",
                    body: formData,
                    credentials: 'omit'
                });
                console.log('Got a response from filter service')

                if (!response.ok) {
                    console.log('Response Not OK')
                    const errorText = await response.text();
                    throw new Error(errorText);
                }

                const result = await response.json();
                // console.log(result)
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
                console.log('received responze');
            } catch (error) {
                alert('FILTER SERVICE FAILURE');
                console.log("Error occured during API call to FILTER_SERVICE: ", error);
            }
        }

        applyInferenceEngineFilters();

    }

    function FiltersDivs() {

        return (
            <>
                <div className="flex flex-wrap gap-10">
                    {Object.entries(filterImages).map(([filter, image]) => (
                        <div key={filter}>
                        <img
                            src={image}
                            alt={filter}
                            className="w-auto h-auto max-w-84 max-h-84 object-cover rounded-lg border"
                        />
                        </div>
                    ))}
                </div>
            </>
        )
    } 

    // console.log(import.meta.env);

    return (
        <>
            {/* <div className="p-4 m-4">Apply Fillters route</div> */}
            <div className="flex space-x-4">
                <div className="m-4 flex-1 text-center">
                    <div className="flex flex-col items-center gap-4">
                        <img
                            src={image}
                            alt="Preview"
                            className="w-auto h-auto max-w-512 max-h-128 object-cover rounded-lg border"
                        />
                        <label htmlFor="image-upload"
                        className="px-4 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600" >
                            Upload Image
                        </label>
                        <input id="image-upload" type="file" accept="image/*" className="hidden" onChange={handleImageChange}/>
                    </div>
                </div>
                <div className="m-4 flex-1 text-center"><FiltersDivs /></div>
            </div>
        </>
    )
}

export default ApplyFilters