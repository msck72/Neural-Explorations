import { useState, useRef } from "react";

const VITE_VIDEO_PROCESSING_SERVICE_API = import.meta.env.VITE_VIDEO_PROCESSING_SERVICE_API;
const VITE_TRACKING_SERVICE_API = import.meta.env.VITE_TRACKING_SERVICE_API;

function VideoManipulations() {
    const [video, setVideo] = useState(null);
    const [fileName, setFileName] = useState("");
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const inputRef = useRef(null);

    const [tracks, setTracks] = useState({});
    const [selected, setSelected] = useState({});

    const handleFile = (file) => {
        if (!file) return;
        if (!file.type.startsWith("video/")) {
            setStatus("error");
            return;
        }

        const videoURL = URL.createObjectURL(file);
        setVideo(videoURL);
        setFileName(file.name);
        setLoading(true);
        setStatus(null);

        const processVideo = async () => {
            if (!VITE_VIDEO_PROCESSING_SERVICE_API) {
                alert("Classification service URL is not configured. Please set VITE_CLASSIFICATION_SERVICE_API.");
                setLoading(false);
                return;
            }

            const formData = new FormData();
            formData.append("video", file);
            const req_id = crypto.randomUUID()
            formData.append("request_id", req_id);
            try {
                const response = await fetch(VITE_VIDEO_PROCESSING_SERVICE_API, {
                    method: "POST",
                    body: formData,
                    credentials: "omit",
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText);
                }
                
                const videoBlob = await response.blob();
                const processedVideoURL = URL.createObjectURL(videoBlob);
                setVideo(processedVideoURL);
                setStatus("success");

                const track_data_req = new FormData();
                track_data_req.append("request_id", req_id)
                const tracking_response = await fetch(VITE_TRACKING_SERVICE_API, {
                    method: "POST",
                    body: track_data_req,
                    credentials: "omit",
                });
                
                if (!tracking_response.ok) {
                    const errorText = await tracking_response.text();
                    throw new Error(errorText);
                }
                
                const tracking_result = await tracking_response.json()
                console.log(tracking_result)
                setTracks(tracking_result)


            } catch (error) {
                setStatus("error");
                console.log("Error occurred during API call to Video Processing service: ", error);
            } finally {
                setLoading(false);
            }
        };

        processVideo();
    };

    const handleVideoChange = (e) => {
        const file = e.target.files[0];
        handleFile(file);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        handleFile(file);
    };

    const handleCheckbox = (e) => {
        const {value, checked} = e.target;
        console.log(value);
        console.log(checked)
        
        setSelected(prev => ({
            ...prev,
            [value]: checked,
        }));
    }

    const handleSubmit = () => {
        console.log(selected)
        
        const checkedItems = {}

        for (const [k, v] of Object.entries(selected)){
            if(v){
                checkedItems[k] = tracks[k];
            }
        }
        
        console.log(checkedItems)
    }

    return (
        <div className="w-full max-w-md mx-auto flex flex-col items-center gap-4 p-6">
            <div
                onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                className={`relative w-full aspect-video rounded-2xl overflow-hidden flex items-center justify-center
                    border-2 border-dashed transition-all duration-200 cursor-pointer
                    ${isDragging
                        ? "border-blue-500 bg-blue-50 scale-[1.01]"
                        : "border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-gray-100"
                    }`}
            >
                {video ? (
                    <video
                        src={video}
                        controls
                        className="w-full h-full object-contain bg-black"
                        onClick={(e) => e.stopPropagation()}
                    />
                ) : (
                    <div className="flex flex-col items-center gap-2 text-gray-400 pointer-events-none select-none">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="w-10 h-10"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={1.5}
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                            />
                        </svg>
                        <p className="text-sm font-medium text-gray-500">
                            Drag & drop a video, or click to browse
                        </p>
                    </div>
                )}

                {loading && (
                    <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center gap-3">
                        <div className="w-8 h-8 border-4 border-white/30 border-t-white rounded-full animate-spin" />
                        <p className="text-white text-sm font-medium">Processing video…</p>
                    </div>
                )}
            </div>

            {fileName && (
                <p className="text-xs text-gray-500 truncate max-w-full">
                    {fileName}
                </p>
            )}

            {status === "success" && (
                <div className="flex items-center gap-2 text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-2 w-full">
                    <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                    Video processed successfully
                </div>
            )}

            {status === "error" && (
                <div className="flex items-center gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2 w-full">
                    <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    Something went wrong processing this video
                </div>
            )}

            <label
                htmlFor="video-upload"
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white text-sm font-medium
                    rounded-lg cursor-pointer shadow-sm hover:bg-blue-700 hover:shadow-md
                    active:scale-[0.98] transition-all duration-150"
            >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
                {video ? "Upload a different video" : "Upload video"}
            </label>

            <input
                ref={inputRef}
                id="video-upload"
                type="file"
                accept="video/*"
                onChange={handleVideoChange}
                className="hidden"
            />

            <div className="flex flex-wrap gap-4">
                {Object.entries(tracks).map(([key, value]) => (
                    <label
                        key={key}
                        className="border p-4 rounded"
                        style={{ display: "block" }}
                    >   
                        <input
                            type="checkbox"
                            value={key}
                            onChange={handleCheckbox}
                        />
                        <h3>{key}</h3>
                        <p>{value.cls}</p>
                    </label>
                ))}
            </div>

            <button onClick={handleSubmit}>
                Submit
            </button>
        </div>
    );
}

export default VideoManipulations;