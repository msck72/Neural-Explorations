import { useState } from "react"
import defaultImage from "../assets/elephant.jpeg"

function Classification() {

    const [image, setImage] = useState(defaultImage)
    const handleImageChange = (e) => {
        const file = e.target.files[0];

        if(!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);
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
            </div>
        </>
    )
}
export default Classification