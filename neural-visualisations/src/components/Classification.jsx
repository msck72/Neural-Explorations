import { useState } from "react"
import defaultImage from "../assets/elephant.jpeg"

const VITE_CLASSIFICATION_SERVICE_API = import.meta.env.VITE_CLASSIFICATION_SERVICE_API;

function Classification() {

    const [image, setImage] = useState(defaultImage)
    const [imageClass, setImageClass] = useState("Cat");
    const [loading, setLoading] = useState(false);

    const handleImageChange = (e) => {
        const file = e.target.files[0];

        if(!file) return;

        const imageUrl = URL.createObjectURL(file);
        setImage(imageUrl);
        setLoading(true);

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
            } finally{
                setLoading(false);
            }
        }

        classifyImage();
    }


    return (
        <section className="px-4 sm:px-6 lg:px-8 text-slate-100">
            <div className="mx-auto flex w-full max-w-6xl flex-col">
                <div className="rounded-4xl border border-white/10 bg-white/5 p-8 shadow-2xl shadow-slate-950/30 backdrop-blur-xl">
                    <div className="mb-8 max-w-2xl">
                        <p className="text-sm uppercase tracking-[0.35em] text-cyan-300">Image classification</p>
                        <p className="mt-3 text-slate-300">Choose an image, then the service will classify it and display the predicted label instantly.</p>
                    </div>

                    <div className="grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
                        <div className="overflow-hidden rounded-4xl border border-white/10 bg-slate-900/80 p-4 shadow-inner shadow-slate-950/20">
                            <div className="relative aspect-4/3 w-full overflow-hidden rounded-3xl border border-white/10 bg-slate-800">
                                <img
                                    src={image}
                                    alt="Preview"
                                    className="h-full w-full object-cover transition duration-500 ease-out"
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
                        </div>

                        <div className="flex flex-col justify-between rounded-4xl border border-white/10 bg-slate-900/80 p-6 shadow-xl shadow-slate-950/30">
                            <div className="space-y-6">
                                <div>
                                    <p className="text-sm uppercase tracking-[0.3em] text-slate-400">Result</p>
                                    <div className="mt-4 inline-flex items-center gap-3 rounded-full bg-slate-950/80 px-5 py-4 text-lg font-semibold text-cyan-100 shadow-sm shadow-cyan-500/10">
                                        <span className="inline-block h-3 w-3 rounded-full bg-cyan-400" />
                                        {imageClass}
                                    </div>
                                </div>

                            </div>

                            <label className="mt-6 inline-flex cursor-pointer items-center justify-center rounded-full bg-cyan-500 px-6 py-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-950 shadow-lg shadow-cyan-500/20 transition duration-200 hover:bg-cyan-400">
                                Upload Image
                                <input type="file" accept="image/*" className="sr-only" onChange={handleImageChange} />
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}
export default Classification