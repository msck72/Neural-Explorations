import { useState } from "react"
import defaultImage from "../assets/elephant.jpeg"

const VITE_CLASSIFICATION_SERVICE_API = import.meta.env.VITE_CLASSIFICATION_SERVICE_API;

function Classification() {

    const [image, setImage] = useState(defaultImage)
    const [imageClass, setImageClass] = useState("Cat");

    const handleImageChange = (e) => {
        const file = e.target.files[0];

        if(!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);

        const classifyImage = async () => {
            if (!VITE_CLASSIFICATION_SERVICE_API) {
                alert('Classification service URL is not configured. Please set VITE_CLASSIFICATION_SERVICE_API.');
                return;
            }

            const formData = new FormData();
            formData.append("image", file);
            try{
                const response = await fetch(VITE_CLASSIFICATION_SERVICE_API, {
                    method: "POST",
                    body: formData,
                    credentials: 'omit'
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText);
                }

                const result = await response.json();
                setImageClass(result.imageClass);
                console.log('received responze');
            } catch (error) {
                alert('Classification service failure');
                console.log("Error occured during API call to classification service failure: ", error);
            }
        }

        classifyImage();
    }


    return (
        <>
            <div>
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
                <div>{imageClass}</div>
            </div>
        </>
    )
}
export default Classification