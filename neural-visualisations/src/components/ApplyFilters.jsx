import { useState } from "react"
import defaultImage from "../assets/elephant.jpeg"

function FiltersDivs() {
    const filters = [
        "filter1",
        "filter2",
        "filter3",
        "filter4",
        "filter4",
        "filter4",
    ]

    return (
        <>
            <div className="flex flex-wrap gap-4">
                {filters.map((filter) => (
                    <div key={filter}>
                        <img
                            src={defaultImage}
                            alt="Preview"
                            className="w-auto h-auto max-w-64 max-h-64 object-cover rounded-lg border"
                        />
                        {filter}
                    </div>
                ))}
            </div>
        </>
    )
}

function ApplyFilters() {

    const [image, setImage] = useState(defaultImage);
    
    const handleImageChange = (e) => {
        const file = e.target.files[0];

        if(!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);
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
                        <input id="image-upload" type="file" accept="image/*" className="hidden"/>
                    </div>
                </div>
                <div className="m-4 flex-1 text-center"><FiltersDivs /></div>
            </div>
        </>
    )
}

export default ApplyFilters