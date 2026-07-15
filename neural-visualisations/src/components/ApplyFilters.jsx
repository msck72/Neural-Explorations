import { useState } from "react"
import defaultImage from "../assets/elephant.jpeg"

const FILTER_SERVICE_API = import.meta.env.VITE_FETCH_SERVICE_API;    

function ApplyFilters() {

    const [image, setImage] = useState(defaultImage);

    const [filterImages, setFilterImages] = useState({
            filter1: defaultImage,
            filter2: defaultImage,
            filter3: defaultImage,
            filter4: defaultImage,
            filter5: defaultImage,
            filter6: defaultImage
        });
    
    const handleImageChange = (e) => {
        const file = e.target.files[0];

        if(!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);
        
        const applyInferenceEngineFilters = async () => {
            const formData = new FormData();
            formData.append("image", file);

            try{
                const response = await fetch(FILTER_SERVICE_API, {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();
                setFilterImages(prev => ({
                    ...prev,
                    ...result,
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